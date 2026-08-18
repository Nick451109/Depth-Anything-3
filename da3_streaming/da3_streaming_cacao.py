#!/usr/bin/env python3

from __future__ import annotations

import time
import argparse
import csv
import gc
import json
import os
import re
import shutil
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.config_utils import load_config
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)

from depth_anything_3.api import DepthAnything3


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False

    if isinstance(depth, np.ndarray):
        input_is_numpy = True
        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    n, h, w = depth_tensor.shape
    tensor_device = depth_tensor.device

    u = torch.arange(w, device=tensor_device).float().view(1, 1, w, 1).expand(n, h, w, 1)
    v = torch.arange(h, device=tensor_device).float().view(1, h, 1, 1).expand(n, h, w, 1)
    ones = torch.ones((n, h, w, 1), device=tensor_device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(n, 4, 4, device=tensor_device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


def remove_duplicates(data_list):
    """Remove duplicated loop-pair entries."""
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        if key not in seen:
            seen[key] = True
            result.append(item)

    return result


class DA3_Streaming:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.overlap_s = 0
        self.overlap_e = self.overlap - self.overlap_s
        self.conf_threshold = 1.5
        self.seed = 42

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.img_dir = image_dir
        self.img_list = None
        self.input_hw = None
        self.processed_hw = None
        self.output_dir = save_dir

        model_config = self.config["Model"]
        self.use_ray_pose = bool(model_config.get("use_ray_pose", False))
        self.save_unaligned_depth = bool(model_config.get("save_unaligned_depth", True))
        self.save_metric_diagnostics = bool(model_config.get("save_metric_diagnostics", True))

        self.metric_diagnostics_path = os.path.join(save_dir, "metric_diagnostics.csv")
        self.alignment_diagnostics_path = os.path.join(save_dir, "alignment_diagnostics.csv")

        self.output_at_input_resolution = model_config.get(
            "output_at_input_resolution", True
        )
        self.save_processed_results = model_config.get("save_processed_results", True)
        self.frame_index_width = int(model_config.get("frame_index_width", 6))

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.result_output_dir = os.path.join(save_dir, "results_output")
        self.pcd_dir = os.path.join(save_dir, "pcd")

        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        print("\n========== DA3 CONFIGURATION ==========")
        print("device:", self.device)
        print("dtype:", self.dtype)
        print("process_res:", model_config.get("process_res", 504))
        print(
            "process_res_method:",
            model_config.get("process_res_method", "upper_bound_resize"),
        )
        print("chunk_size:", self.chunk_size)
        print("overlap:", self.overlap)
        print("use_ray_pose:", self.use_ray_pose)
        print("ref_view_strategy:", model_config.get("ref_view_strategy", "middle"))
        print("align_method:", model_config.get("align_method", "sim3"))
        print("save_unaligned_depth:", self.save_unaligned_depth)
        print("save_metric_diagnostics:", self.save_metric_diagnostics)
        print("=======================================\n")

        print("Loading model...")

        with open(self.config["Weights"]["DA3_CONFIG"], encoding="utf-8") as file:
            da3_config = json.load(file)

        self.model = DepthAnything3(**da3_config)
        weight = load_file(self.config["Weights"]["DA3"])
        self.model.load_state_dict(weight, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)

        self.skyseg_session = None
        self.chunk_indices = None
        self.loop_list = []
        self.loop_optimizer = Sim3LoopOptimizer(self.config)
        self.sim3_list = []
        self.loop_sim3_list = []
        self.loop_predict_list = []
        self.loop_enable = self.config["Model"]["loop_enable"]

        if self.loop_enable:
            loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
            self.loop_detector = LoopDetector(
                image_dir=image_dir,
                output=loop_info_save_path,
                config=self.config,
            )
            self.loop_detector.load_model()

        print("init done.")

    @staticmethod
    def _natural_sort_key(path):
        """Sort frame paths numerically when their names contain numbers."""
        name = Path(path).name
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)
        ]

    @staticmethod
    def _read_rgb(image_path):
        """Load an image without changing its original RGB raster."""
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _normalize_frame_maps(array, expected_frames, name):
        """Normalize DA3 map outputs to [N, H, W] without collapsing N=1."""
        if isinstance(array, torch.Tensor):
            result = array.detach().cpu().numpy()
        else:
            result = np.asarray(array)

        if result.ndim == 4 and result.shape[0] == 1:
            result = result[0]
        if result.ndim == 2 and expected_frames == 1:
            result = result[None, ...]
        if result.ndim != 3 or result.shape[0] != expected_frames:
            raise ValueError(
                f"Unexpected {name} shape {result.shape}; expected [N, H, W] "
                f"with N={expected_frames}."
            )
        return result.astype(np.float32, copy=False)

    @staticmethod
    def _scale_intrinsics(intrinsics, source_hw, target_hw):
        """Scale camera intrinsics between raster resolutions."""
        source_h, source_w = source_hw
        target_h, target_w = target_hw
        scaled = np.asarray(intrinsics, dtype=np.float32).copy()
        scaled[0, :] *= target_w / float(source_w)
        scaled[1, :] *= target_h / float(source_h)
        scaled[2, :] = np.asarray(intrinsics, dtype=np.float32)[2, :]
        return scaled

    @staticmethod
    def _to_numpy_cpu(value):
        """Convert a torch Tensor or array-like value to NumPy on CPU."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _scale_to_float(cls, value):
        """Convert a Sim(3) scale value to a Python float."""
        array = cls._to_numpy_cpu(value).reshape(-1)
        if array.size == 0:
            raise ValueError("Empty alignment scale.")
        return float(array[0])

    @staticmethod
    def _append_csv_row(path, fieldnames, row):
        """Append one row to a CSV file, creating the header if necessary."""
        file_exists = os.path.exists(path)
        file_is_empty = (not file_exists) or os.path.getsize(path) == 0

        with open(path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if file_is_empty:
                writer.writeheader()
            writer.writerow(row)

    def _append_metric_diagnostics(self, predictions, chunk_idx, global_indices):
        """
        Save raw DA3 depth/intrinsic diagnostics BEFORE any inter-chunk Sim(3).
        """
        if not self.save_metric_diagnostics:
            return

        fieldnames = [
            "chunk_idx",
            "frame_index",
            "source_filename",
            "use_ray_pose",
            "processed_width",
            "processed_height",
            "fx",
            "fy",
            "cx",
            "cy",
            "f_avg",
            "fov_x_deg",
            "fov_y_deg",
            "depth_min",
            "depth_p05",
            "depth_median",
            "depth_p95",
            "depth_max",
            "confidence_median",
        ]

        intrinsics_all = self._to_numpy_cpu(predictions.intrinsics)

        for local_idx, global_idx in enumerate(global_indices):
            depth = np.asarray(predictions.depth[local_idx], dtype=np.float32)
            confidence = np.asarray(predictions.conf[local_idx], dtype=np.float32)
            k_matrix = np.asarray(intrinsics_all[local_idx], dtype=np.float32)

            height, width = depth.shape
            fx = float(k_matrix[0, 0])
            fy = float(k_matrix[1, 1])
            cx = float(k_matrix[0, 2])
            cy = float(k_matrix[1, 2])
            f_avg = 0.5 * (fx + fy)

            fov_x_deg = (
                float(np.degrees(2.0 * np.arctan(width / (2.0 * fx))))
                if fx > 0
                else np.nan
            )
            fov_y_deg = (
                float(np.degrees(2.0 * np.arctan(height / (2.0 * fy))))
                if fy > 0
                else np.nan
            )

            valid_depth = depth[np.isfinite(depth) & (depth > 0)]
            if valid_depth.size > 0:
                depth_min = float(np.min(valid_depth))
                depth_p05 = float(np.percentile(valid_depth, 5))
                depth_median = float(np.median(valid_depth))
                depth_p95 = float(np.percentile(valid_depth, 95))
                depth_max = float(np.max(valid_depth))
            else:
                depth_min = np.nan
                depth_p05 = np.nan
                depth_median = np.nan
                depth_p95 = np.nan
                depth_max = np.nan

            valid_conf = confidence[np.isfinite(confidence)]
            confidence_median = (
                float(np.median(valid_conf)) if valid_conf.size > 0 else np.nan
            )

            source_filename = Path(self.img_list[global_idx]).name

            row = {
                "chunk_idx": chunk_idx,
                "frame_index": global_idx,
                "source_filename": source_filename,
                "use_ray_pose": self.use_ray_pose,
                "processed_width": width,
                "processed_height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "f_avg": f_avg,
                "fov_x_deg": fov_x_deg,
                "fov_y_deg": fov_y_deg,
                "depth_min": depth_min,
                "depth_p05": depth_p05,
                "depth_median": depth_median,
                "depth_p95": depth_p95,
                "depth_max": depth_max,
                "confidence_median": confidence_median,
            }

            self._append_csv_row(self.metric_diagnostics_path, fieldnames, row)

    def _append_alignment_diagnostic(self, source_chunk, target_chunk, s, r_matrix, t_vector):
        """Save the Sim(3) estimated between two consecutive chunks."""
        if not self.save_metric_diagnostics:
            return

        scale = self._scale_to_float(s)
        rotation = self._to_numpy_cpu(r_matrix).reshape(3, 3)
        translation = self._to_numpy_cpu(t_vector).reshape(-1)

        fieldnames = [
            "source_chunk",
            "target_chunk",
            "scale",
            "translation_x",
            "translation_y",
            "translation_z",
            "rotation_determinant",
        ]

        row = {
            "source_chunk": source_chunk,
            "target_chunk": target_chunk,
            "scale": scale,
            "translation_x": float(translation[0]),
            "translation_y": float(translation[1]),
            "translation_z": float(translation[2]),
            "rotation_determinant": float(np.linalg.det(rotation)),
        }

        self._append_csv_row(self.alignment_diagnostics_path, fieldnames, row)

    def _validate_input_resolution(self):
        """Require a constant resolution so every output remains registered."""
        dimensions = []
        for image_path in self.img_list:
            with Image.open(image_path) as image:
                dimensions.append((image.height, image.width))

        unique_dimensions = sorted(set(dimensions))
        if len(unique_dimensions) != 1:
            raise ValueError(
                "All frames must have the same resolution for registered output. "
                f"Found: {unique_dimensions}"
            )

        self.input_hw = unique_dimensions[0]
        print(f"Input resolution: {self.input_hw[1]}x{self.input_hw[0]}")

    def _restore_map_to_input_resolution(self, value):
        """Resize a continuous map to the original input raster."""
        input_h, input_w = self.input_hw
        return cv2.resize(
            np.asarray(value, dtype=np.float32),
            (input_w, input_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)

    def get_loop_pairs(self):
        self.loop_detector.run()
        return self.loop_detector.get_loop_list()

    def save_depth_conf_result(self, predictions, chunk_idx, s, r_matrix, t_vector):
        """
        Save final per-frame NPZ files.

        depth_unaligned: raw DA3 depth before inter-chunk Sim(3) scale.
        depth:           depth after applying the accumulated Sim(3) scale.
        """
        if not self.config["Model"]["save_depth_conf_result"]:
            return

        os.makedirs(self.result_output_dir, exist_ok=True)
        alignment_scale = self._scale_to_float(s)

        chunk_start, chunk_end = self.chunk_indices[chunk_idx]
        chunk_length = chunk_end - chunk_start

        if len(self.chunk_indices) == 1:
            save_indices = list(range(chunk_length))
        elif chunk_idx == 0:
            save_indices = list(range(0, chunk_length - self.overlap_e))
        elif chunk_idx == len(self.chunk_indices) - 1:
            save_indices = list(range(self.overlap_s, chunk_length))
        else:
            save_indices = list(
                range(self.overlap_s, chunk_length - self.overlap_e)
            )

        print("[save_depth_conf_result] save_indices:")

        intrinsics_all = self._to_numpy_cpu(predictions.intrinsics)
        extrinsics_all = self._to_numpy_cpu(predictions.extrinsics)

        for local_idx in save_indices:
            global_idx = chunk_start + local_idx
            print(f"{global_idx}, ", end="")

            source_path = os.path.realpath(self.img_list[global_idx])
            image_original = self._read_rgb(source_path)

            image_processed = np.asarray(
                predictions.processed_images[local_idx], dtype=np.uint8
            )

            depth_processed_unaligned = np.asarray(
                predictions.depth[local_idx], dtype=np.float32
            ).copy()

            depth_processed = (
                depth_processed_unaligned * alignment_scale
            ).astype(np.float32, copy=False)

            conf_processed = np.maximum(
                np.asarray(predictions.conf[local_idx], dtype=np.float32),
                0.0,
            )

            intrinsics_processed = np.asarray(
                intrinsics_all[local_idx], dtype=np.float32
            )

            processed_hw = tuple(depth_processed.shape)
            input_hw = tuple(image_original.shape[:2])

            if processed_hw != tuple(image_processed.shape[:2]):
                raise ValueError(
                    "Processed RGB and depth shapes do not match: "
                    f"{image_processed.shape[:2]} vs {processed_hw}."
                )

            if input_hw != self.input_hw:
                raise ValueError(
                    f"Unexpected input resolution for {source_path}: {input_hw}; "
                    f"expected {self.input_hw}."
                )

            if self.output_at_input_resolution:
                image = image_original
                depth = self._restore_map_to_input_resolution(depth_processed)
                depth_unaligned = self._restore_map_to_input_resolution(
                    depth_processed_unaligned
                )
                conf = np.maximum(
                    self._restore_map_to_input_resolution(conf_processed), 0.0
                )
                intrinsics = self._scale_intrinsics(
                    intrinsics_processed, processed_hw, input_hw
                )
            else:
                image = image_processed
                depth = depth_processed
                depth_unaligned = depth_processed_unaligned
                conf = conf_processed
                intrinsics = intrinsics_processed

            filename = f"frame_{global_idx:0{self.frame_index_width}d}.npz"
            filepath = os.path.join(self.result_output_dir, filename)

            payload = {
                "image": image,
                "depth": depth,
                "conf": conf,
                "intrinsics": intrinsics,
                "extrinsics": np.asarray(
                    extrinsics_all[local_idx], dtype=np.float32
                ),
                "frame_index": np.int32(global_idx),
                "source_filename": np.asarray(os.path.basename(source_path)),
                "source_path": np.asarray(source_path),
                "input_hw": np.asarray(input_hw, dtype=np.int32),
                "processed_hw": np.asarray(processed_hw, dtype=np.int32),
                "process_res": np.int32(
                    self.config["Model"].get("process_res", 504)
                ),
                "process_res_method": np.asarray(
                    self.config["Model"].get(
                        "process_res_method", "upper_bound_resize"
                    )
                ),
                "alignment_scale": np.float32(alignment_scale),
                "use_ray_pose": np.asarray(self.use_ray_pose),
            }

            if self.save_unaligned_depth:
                payload["depth_unaligned"] = depth_unaligned

            if self.save_processed_results:
                payload.update(
                    {
                        "image_processed": image_processed,
                        "depth_processed": depth_processed,
                        "conf_processed": conf_processed,
                        "intrinsics_processed": intrinsics_processed,
                    }
                )

                if self.save_unaligned_depth:
                    payload["depth_processed_unaligned"] = (
                        depth_processed_unaligned
                    )

            if self.config["Model"]["save_debug_info"]:
                payload.update(
                    {
                        "alignment_rotation": self._to_numpy_cpu(r_matrix),
                        "alignment_translation": self._to_numpy_cpu(t_vector),
                    }
                )

            np.savez_compressed(filepath, **payload)

        print("")

    def process_single_chunk(
        self,
        range_1,
        chunk_idx=None,
        range_2=None,
        is_loop=False,
    ):
        range_1_start, range_1_end = range_1
        chunk_image_paths = list(self.img_list[range_1_start:range_1_end])

        if range_2 is not None:
            range_2_start, range_2_end = range_2
            chunk_image_paths += self.img_list[range_2_start:range_2_end]

        print(f"Loaded {len(chunk_image_paths)} images")

        ref_view_strategy = self.config["Model"][
            "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"
        ]

        process_res = self.config["Model"].get("process_res", 504)
        process_res_method = self.config["Model"].get(
            "process_res_method", "upper_bound_resize"
        )

        print("\n========== CHUNK INFERENCE ==========")
        print("chunk_idx:", chunk_idx)
        print("range_1:", range_1)
        print("range_2:", range_2)
        print("number of images:", len(chunk_image_paths))
        print("process_res:", process_res)
        print("process_res_method:", process_res_method)
        print("ref_view_strategy:", ref_view_strategy)
        print("use_ray_pose:", self.use_ray_pose)
        print("=====================================\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        amp_context = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device == "cuda"
            else nullcontext()
        )

        with torch.no_grad():
            with amp_context:
                predictions = self.model.inference(
                    chunk_image_paths,
                    ref_view_strategy=ref_view_strategy,
                    use_ray_pose=self.use_ray_pose,
                    process_res=process_res,
                    process_res_method=process_res_method,
                )

        expected_frames = len(chunk_image_paths)
        predictions.depth = self._normalize_frame_maps(
            predictions.depth, expected_frames, "depth"
        )
        predictions.conf = np.maximum(
            self._normalize_frame_maps(
                predictions.conf, expected_frames, "confidence"
            )
            - 1.0,
            0.0,
        ).astype(np.float32, copy=False)

        current_processed_hw = tuple(predictions.depth.shape[-2:])
        if self.processed_hw is None:
            self.processed_hw = current_processed_hw
        elif self.processed_hw != current_processed_hw:
            raise ValueError(
                "DA3 produced inconsistent processed resolutions: "
                f"{self.processed_hw} and {current_processed_hw}."
            )

        print("processed_images:", predictions.processed_images.shape)
        print("depth:", predictions.depth.shape)
        print("conf:", predictions.conf.shape)
        print("extrinsics:", predictions.extrinsics.shape)
        print("intrinsics:", predictions.intrinsics.shape)

        if not is_loop and range_2 is None and chunk_idx is not None:
            global_indices = list(range(range_1_start, range_1_end))
            self._append_metric_diagnostics(
                predictions,
                chunk_idx,
                global_indices,
            )
            print(
                f"[DEBUG] Raw DA3 metrics saved to {self.metric_diagnostics_path}"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_loop:
            save_dir = self.result_loop_dir
            filename = (
                f"loop_{range_1[0]}_{range_1[1]}_"
                f"{range_2[0]}_{range_2[1]}.npy"
            )
        else:
            if chunk_idx is None:
                raise ValueError(
                    "chunk_idx must be provided when is_loop is False"
                )
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions.extrinsics
            intrinsics = predictions.intrinsics
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        np.save(save_path, predictions)
        return predictions

    def get_chunk_indices(self):
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                chunk_indices.append((start_idx, end_idx))

        return chunk_indices, num_chunks

    def align_2pcds(
        self,
        point_map1,
        conf1,
        point_map2,
        conf2,
        chunk1_depth,
        chunk2_depth,
        chunk1_depth_conf,
        chunk2_depth_conf,
    ):
        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            (
                scale_factor_return,
                quality_score,
                method_used,
            ) = precompute_scale_chunks_with_depth(
                chunk1_depth,
                chunk1_depth_conf,
                chunk2_depth,
                chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            print(
                "[Depth Scale Precompute] "
                f"scale: {scale_factor_return}, "
                f"quality_score: {quality_score}, "
                f"method_used: {method_used}"
            )
            scale_factor = scale_factor_return

        s, r_matrix, t_vector = weighted_align_point_maps(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )

        print("Estimated Scale:", s)
        print("Estimated Rotation:\n", r_matrix)
        print("Estimated Translation:", t_vector)

        return s, r_matrix, t_vector

    def get_loop_sim3_from_loop_predict(self, loop_predict_list):
        loop_sim3_list = []

        for item in loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop_org = depth_to_point_cloud_vectorized(
                item[1].depth,
                item[1].intrinsics,
                item[1].extrinsics,
            )

            chunk_a_s = 0
            chunk_a_e = chunk_a_len = chunk_a_range[1] - chunk_a_range[0]
            chunk_b_s = -chunk_b_range[1] + chunk_b_range[0]
            chunk_b_e = point_map_loop_org.shape[0]
            chunk_b_len = chunk_b_range[1] - chunk_b_range[0]

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_len
            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_len

            print("chunk_a align")

            point_map_loop_a = point_map_loop_org[chunk_a_s:chunk_a_e]
            conf_loop = item[1].conf[chunk_a_s:chunk_a_e]
            print(self.chunk_indices[chunk_idx_a])
            print(chunk_a_range)
            print(chunk_a_rela_begin, chunk_a_rela_end)

            chunk_data_a = np.load(
                os.path.join(
                    self.result_unaligned_dir,
                    f"chunk_{chunk_idx_a}.npy",
                ),
                allow_pickle=True,
            ).item()

            point_map_a = depth_to_point_cloud_vectorized(
                chunk_data_a.depth,
                chunk_data_a.intrinsics,
                chunk_data_a.extrinsics,
            )
            point_map_a = point_map_a[chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_a_depth = np.squeeze(
                    chunk_data_a.depth[chunk_a_rela_begin:chunk_a_rela_end]
                )
                chunk_a_depth_conf = np.squeeze(
                    chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]
                )
                chunk_a_loop_depth = np.squeeze(item[1].depth[chunk_a_s:chunk_a_e])
                chunk_a_loop_depth_conf = np.squeeze(item[1].conf[chunk_a_s:chunk_a_e])
            else:
                chunk_a_depth = None
                chunk_a_loop_depth = None
                chunk_a_depth_conf = None
                chunk_a_loop_depth_conf = None

            s_a, r_a, t_a = self.align_2pcds(
                point_map_a,
                conf_a,
                point_map_loop_a,
                conf_loop,
                chunk_a_depth,
                chunk_a_loop_depth,
                chunk_a_depth_conf,
                chunk_a_loop_depth_conf,
            )

            print("chunk_b align")

            point_map_loop_b = point_map_loop_org[chunk_b_s:chunk_b_e]
            conf_loop = item[1].conf[chunk_b_s:chunk_b_e]
            print(self.chunk_indices[chunk_idx_b])
            print(chunk_b_range)
            print(chunk_b_rela_begin, chunk_b_rela_end)

            chunk_data_b = np.load(
                os.path.join(
                    self.result_unaligned_dir,
                    f"chunk_{chunk_idx_b}.npy",
                ),
                allow_pickle=True,
            ).item()

            point_map_b = depth_to_point_cloud_vectorized(
                chunk_data_b.depth,
                chunk_data_b.intrinsics,
                chunk_data_b.extrinsics,
            )
            point_map_b = point_map_b[chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_b_depth = np.squeeze(
                    chunk_data_b.depth[chunk_b_rela_begin:chunk_b_rela_end]
                )
                chunk_b_depth_conf = np.squeeze(
                    chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]
                )
                chunk_b_loop_depth = np.squeeze(item[1].depth[chunk_b_s:chunk_b_e])
                chunk_b_loop_depth_conf = np.squeeze(item[1].conf[chunk_b_s:chunk_b_e])
            else:
                chunk_b_depth = None
                chunk_b_loop_depth = None
                chunk_b_depth_conf = None
                chunk_b_loop_depth_conf = None

            s_b, r_b, t_b = self.align_2pcds(
                point_map_b,
                conf_b,
                point_map_loop_b,
                conf_loop,
                chunk_b_depth,
                chunk_b_loop_depth,
                chunk_b_depth_conf,
                chunk_b_loop_depth_conf,
            )

            print("a -> b SIM 3")
            s_ab, r_ab, t_ab = compute_sim3_ab(
                (s_a, r_a, t_a),
                (s_b, r_b, t_b),
            )
            print("Estimated Scale:", s_ab)
            print("Estimated Rotation:\n", r_ab)
            print("Estimated Translation:", t_ab)

            loop_sim3_list.append(
                (chunk_idx_a, chunk_idx_b, (s_ab, r_ab, t_ab))
            )

        return loop_sim3_list

    def plot_loop_closure(
        self,
        input_abs_poses,
        optimized_abs_poses,
        save_name="sim3_opt_result.png",
    ):
        def extract_xyz(pose_tensor):
            poses = pose_tensor.cpu().numpy()
            return poses[:, 0], poses[:, 1], poses[:, 2]

        x0, _, y0 = extract_xyz(input_abs_poses)
        x1, _, y1 = extract_xyz(optimized_abs_poses)

        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
        plt.plot(x1, y1, "o-", label="After Optimization")

        for i, j, _ in self.loop_sim3_list:
            plt.plot(
                [x0[i], x0[j]],
                [y0[i], y0[j]],
                "r--",
                alpha=0.25,
                label="Loop (Before)" if i == 5 else "",
            )
            plt.plot(
                [x1[i], x1[j]],
                [y1[i], y1[j]],
                "g-",
                alpha=0.25,
                label="Loop (After)" if i == 5 else "",
            )

        plt.gca().set_aspect("equal")
        plt.title("Sim3 Loop Closure Optimization")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) "
                f"must be less than chunk size ({self.chunk_size})"
            )

        self.chunk_indices, num_chunks = self.get_chunk_indices()

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} "
            f"chunks of size {self.chunk_size} with {self.overlap} overlap"
        )
        print("Chunk ranges:", self.chunk_indices)

        pre_predictions = None

        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)}")

            cur_predictions = self.process_single_chunk(
                self.chunk_indices[chunk_idx],
                chunk_idx=chunk_idx,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if chunk_idx > 0:
                print(
                    f"Aligning {chunk_idx - 1} and {chunk_idx} "
                    f"(Total {len(self.chunk_indices) - 1})"
                )

                chunk_data1 = pre_predictions
                chunk_data2 = cur_predictions

                point_map1 = depth_to_point_cloud_vectorized(
                    chunk_data1.depth,
                    chunk_data1.intrinsics,
                    chunk_data1.extrinsics,
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    chunk_data2.depth,
                    chunk_data2.intrinsics,
                    chunk_data2.extrinsics,
                )

                point_map1 = point_map1[-self.overlap :]
                point_map2 = point_map2[: self.overlap]
                conf1 = chunk_data1.conf[-self.overlap :]
                conf2 = chunk_data2.conf[: self.overlap]

                if self.config["Model"]["align_method"] == "scale+se3":
                    chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                    chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                    chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                    chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                s, r_matrix, t_vector = self.align_2pcds(
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    chunk1_depth,
                    chunk2_depth,
                    chunk1_depth_conf,
                    chunk2_depth_conf,
                )

                self._append_alignment_diagnostic(
                    chunk_idx - 1,
                    chunk_idx,
                    s,
                    r_matrix,
                    t_vector,
                )

                self.sim3_list.append((s, r_matrix, t_vector))

            pre_predictions = cur_predictions

        if self.loop_enable:
            self.loop_list = self.get_loop_pairs()
            del self.loop_detector

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Loop SIM(3) estimating...")
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
            loop_results = remove_duplicates(loop_results)
            print(loop_results)

            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1],
                    range_2=item[3],
                    is_loop=True,
                )
                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

            self.loop_sim3_list = self.get_loop_sim3_from_loop_predict(
                self.loop_predict_list
            )

            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )
            self.sim3_list = self.loop_optimizer.optimize(
                self.sim3_list,
                self.loop_sim3_list,
            )
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )

            self.plot_loop_closure(
                input_abs_poses,
                optimized_abs_poses,
                save_name="sim3_opt_result.png",
            )

        if len(self.chunk_indices) == 1:
            print("Single chunk sequence: saving outputs without inter-chunk alignment")

            chunk_data_first = np.load(
                os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                allow_pickle=True,
            ).item()

            points_first = depth_to_point_cloud_vectorized(
                chunk_data_first.depth,
                chunk_data_first.intrinsics,
                chunk_data_first.extrinsics,
            )
            colors_first = chunk_data_first.processed_images
            confs_first = chunk_data_first.conf

            save_confident_pointcloud_batch(
                points=points_first,
                colors=colors_first,
                confs=confs_first,
                output_path=os.path.join(self.pcd_dir, "000000_pcd.ply"),
                conf_threshold=np.mean(confs_first)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

            self.save_depth_conf_result(
                chunk_data_first,
                0,
                1.0,
                np.eye(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )

            self.save_camera_poses()
            print("Done.")
            return

        print("Apply alignment")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)

        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(
                f"Applying {chunk_idx + 1} -> {chunk_idx} "
                f"(Total {len(self.chunk_indices) - 1})"
            )

            s, r_matrix, t_vector = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(
                    self.result_unaligned_dir,
                    f"chunk_{chunk_idx + 1}.npy",
                ),
                allow_pickle=True,
            ).item()

            aligned_chunk_data = {}
            aligned_chunk_data["world_points"] = (
                depth_to_point_cloud_optimized_torch(
                    chunk_data.depth,
                    chunk_data.intrinsics,
                    chunk_data.extrinsics,
                )
            )
            aligned_chunk_data["world_points"] = apply_sim3_direct_torch(
                aligned_chunk_data["world_points"],
                s,
                r_matrix,
                t_vector,
            )
            aligned_chunk_data["conf"] = chunk_data.conf
            aligned_chunk_data["images"] = chunk_data.processed_images

            aligned_path = os.path.join(
                self.result_aligned_dir,
                f"chunk_{chunk_idx + 1}.npy",
            )
            np.save(aligned_path, aligned_chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                    allow_pickle=True,
                ).item()

                np.save(
                    os.path.join(self.result_aligned_dir, "chunk_0.npy"),
                    chunk_data_first,
                )

                points_first = depth_to_point_cloud_vectorized(
                    chunk_data_first.depth,
                    chunk_data_first.intrinsics,
                    chunk_data_first.extrinsics,
                )
                colors_first = chunk_data_first.processed_images
                confs_first = chunk_data_first.conf

                ply_path_first = os.path.join(self.pcd_dir, "000000_pcd.ply")
                save_confident_pointcloud_batch(
                    points=points_first,
                    colors=colors_first,
                    confs=confs_first,
                    output_path=ply_path_first,
                    conf_threshold=np.mean(confs_first)
                    * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                    sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
                )

                if self.config["Model"]["save_depth_conf_result"]:
                    self.save_depth_conf_result(
                        chunk_data_first,
                        0,
                        1.0,
                        np.eye(3, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                    )

            points = self._to_numpy_cpu(
                aligned_chunk_data["world_points"]
            ).reshape(-1, 3)
            colors = aligned_chunk_data["images"].reshape(-1, 3).astype(np.uint8)
            confs = aligned_chunk_data["conf"].reshape(-1)

            ply_path = os.path.join(
                self.pcd_dir,
                f"{chunk_idx + 1:06d}_pcd.ply",
            )

            save_confident_pointcloud_batch(
                points=points,
                colors=colors,
                confs=confs,
                output_path=ply_path,
                conf_threshold=np.mean(confs)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

            if self.config["Model"]["save_depth_conf_result"]:
                # IMPORTANT: do NOT modify chunk_data.depth in-place.
                # save_depth_conf_result stores both raw and aligned depth.
                self.save_depth_conf_result(
                    chunk_data,
                    chunk_idx + 1,
                    s,
                    r_matrix,
                    t_vector,
                )

        self.save_camera_poses()
        print("Done.")

    def run(self):
        print(f"Loading images from {self.img_dir}...")

        valid_extensions = {".jpg", ".jpeg", ".png"}
        self.img_list = sorted(
            [
                str(path)
                for path in Path(self.img_dir).iterdir()
                if path.is_file() and path.suffix.lower() in valid_extensions
            ],
            key=self._natural_sort_key,
        )

        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")

        print(f"Found {len(self.img_list)} images")
        self._validate_input_resolution()
        self.process_long_sequence()

    def save_camera_poses(self):
        """
        Save camera poses and intrinsics from all chunks.
        """
        chunk_colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [128, 0, 0],
            [0, 128, 0],
            [0, 0, 128],
            [128, 128, 0],
        ]

        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]

        first_chunk_extrinsics = self._to_numpy_cpu(first_chunk_extrinsics)
        first_chunk_intrinsics = self._to_numpy_cpu(first_chunk_intrinsics)

        first_chunk_end = (
            first_chunk_range[1]
            if len(self.all_camera_poses) == 1
            else first_chunk_range[1] - self.overlap_e
        )

        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_end)):
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :] = first_chunk_extrinsics[i]
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]

            chunk_extrinsics = self._to_numpy_cpu(chunk_extrinsics)
            chunk_intrinsics = self._to_numpy_cpu(chunk_intrinsics)

            s, r_matrix, t_vector = self.sim3_list[chunk_idx - 1]
            scale = self._scale_to_float(s)
            rotation = self._to_numpy_cpu(r_matrix).reshape(3, 3)
            translation = self._to_numpy_cpu(t_vector).reshape(3)

            sim3_matrix = np.eye(4, dtype=np.float64)
            sim3_matrix[:3, :3] = scale * rotation
            sim3_matrix[:3, 3] = translation

            chunk_range_end = (
                chunk_range[1] - self.overlap_e
                if chunk_idx < len(self.all_camera_poses) - 1
                else chunk_range[1]
            )

            for i, idx in enumerate(
                range(chunk_range[0] + self.overlap_s, chunk_range_end)
            ):
                w2c = np.eye(4, dtype=np.float64)
                w2c[:3, :] = chunk_extrinsics[i + self.overlap_s]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = sim3_matrix @ c2w
                transformed_c2w[:3, :3] /= scale

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i + self.overlap_s]

        if any(pose is None for pose in all_poses):
            raise RuntimeError("Missing camera pose for one or more frames.")

        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w", encoding="utf-8") as file:
            for pose in all_poses:
                flat_pose = pose.flatten()
                file.write(" ".join(str(x) for x in flat_pose) + "\n")

        print(f"Camera poses saved to {poses_path}")

        if any(intrinsic is None for intrinsic in all_intrinsics):
            raise RuntimeError("Missing camera intrinsics for one or more frames.")

        processed_intrinsics = [
            np.asarray(intrinsic, dtype=np.float32)
            for intrinsic in all_intrinsics
        ]

        if self.output_at_input_resolution:
            output_intrinsics = [
                self._scale_intrinsics(
                    intrinsic,
                    self.processed_hw,
                    self.input_hw,
                )
                for intrinsic in processed_intrinsics
            ]
        else:
            output_intrinsics = processed_intrinsics

        def write_intrinsics(path, values):
            with open(path, "w", encoding="utf-8") as file:
                for intrinsic in values:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    file.write(f"{fx} {fy} {cx} {cy}\n")

        intrinsics_path = os.path.join(
            self.output_dir,
            "camera_intrinsics.txt",
        )
        write_intrinsics(intrinsics_path, output_intrinsics)

        legacy_intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
        write_intrinsics(legacy_intrinsics_path, output_intrinsics)

        if self.save_processed_results:
            processed_intrinsics_path = os.path.join(
                self.output_dir,
                "camera_intrinsics_processed.txt",
            )
            write_intrinsics(processed_intrinsics_path, processed_intrinsics)

        print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w", encoding="utf-8") as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {len(all_poses)}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
            file.write("end_header\n")

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                file.write(
                    f"{position[0]} {position[1]} {position[2]} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        """Delete temporary files when configured to do so."""
        if not self.delete_temp_files:
            print("Temporary files preserved for debugging.")
            return

        total_space = 0

        for directory in [
            self.result_unaligned_dir,
            self.result_aligned_dir,
            self.result_loop_dir,
        ]:
            print(f"Deleting temp files under {directory}")
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    total_space += os.path.getsize(file_path)
                    os.remove(file_path)

        print("Deleting temp files done.")
        print(f"Saved disk space: {total_space / 1024 / 1024 / 1024:.4f} GiB")

class DA3RealtimeDepth:
    """
    Inferencia monocular DA3 en tiempo real.

    Modos de profundidad:
        metric_focal:
            Para DA3METRIC-LARGE.
            depth_m = depth_raw * focal_processed / 300.

        metric_direct:
            Para modelos cuya salida ya está en metros,
            por ejemplo DA3NESTED-GIANT-LARGE.

        relative:
            Para DA3-SMALL, BASE, LARGE, MONO, etc.
            No se interpreta la salida como metros.
    """

    VALID_DEPTH_MODES = {
        "metric_focal",
        "metric_direct",
        "relative",
    }

    def __init__(self, config):
        self.config = config

        #warmup ignore for comparison
        self.benchmark_times = []
        self.benchmark_warmup = 20
        self.benchmark_frames = 200
        self.frame_counter = 0

        realtime_config = self.config.get("Realtime", {})

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Precision que DA3 seleccionara para AMP en esta GPU
        if self.device.type == "cuda":
            self.dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

        else:
            self.dtype = torch.float32

        #para testear fp16 forzado
        # if self.device.type == "cuda":
        #     self.dtype = torch.float16
        # else:
        #     self.dtype = torch.float32

        self.model_source = realtime_config.get("model_source")

        if not self.model_source:
            raise ValueError(
                "Realtime.model_source debe estar definido en el YAML."
            )

        self.depth_mode = realtime_config.get(
            "depth_mode",
            "metric_focal",
        )

        if self.depth_mode not in self.VALID_DEPTH_MODES:
            raise ValueError(
                f"depth_mode inválido: {self.depth_mode}. "
                f"Opciones: {sorted(self.VALID_DEPTH_MODES)}"
            )

        self.process_res = int(
            realtime_config.get("process_res", 504)
        )

        self.process_res_method = realtime_config.get(
            "process_res_method",
            "upper_bound_resize",
        )

        self.camera_source = realtime_config.get(
            "camera_source",
            0,
        )

        if (
            isinstance(self.camera_source, str)
            and self.camera_source.isdigit()
        ):
            self.camera_source = int(self.camera_source)

        self.capture_width = int(
            realtime_config.get("capture_width", 1920)
        )

        self.capture_height = int(
            realtime_config.get("capture_height", 1080)
        )

        self.capture_fps = float(
            realtime_config.get("capture_fps", 30.0)
        )

        self.capture_fourcc = str(
            realtime_config.get(
                "capture_fourcc",
                "MJPG",
            )
        )

        self.stereo_enabled = bool(
            realtime_config.get(
                "stereo_enabled",
                False,
            )
        )

        self.stereo_view = str(
            realtime_config.get(
                "stereo_view",
                "left",
            )
        ).lower()

        if self.stereo_view not in {
            "left",
            "right",
        }:
            raise ValueError(
                "Realtime.stereo_view debe ser "
                "'left' o 'right'."
            )

        self._printed_stereo_shape = False

        self.display = bool(
            realtime_config.get("display", True)
        )

        self.display_scale = float(
            realtime_config.get("display_scale", 0.5)
        )

        fx_value = realtime_config.get("fx")
        fy_value = realtime_config.get("fy")

        self.fx = (
            None if fx_value is None else float(fx_value)
        )

        self.fy = (
            None if fy_value is None else float(fy_value)
        )

        self.metric_focal_reference = float(
            realtime_config.get(
                "metric_focal_reference",
                300.0,
            )
        )

        if self.depth_mode == "metric_focal":
            if self.fx is None or self.fy is None:
                raise ValueError(
                    "DA3METRIC-LARGE necesita fx y fy de la cámara "
                    "para producir profundidad métrica."
                )

        print("\n========== DA3 REALTIME ==========")
        print("device:", self.device)
        print("dtype:", self.dtype)
        print("model_source:", self.model_source)
        print("depth_mode:", self.depth_mode)
        print("process_res:", self.process_res)
        print(
            "process_res_method:",
            self.process_res_method,
        )
        print(
            "capture:",
            f"{self.capture_width}x{self.capture_height}",
        )

        if self.fx is not None:
            print("fx:", self.fx)

        if self.fy is not None:
            print("fy:", self.fy)

        print("==================================\n")

        self._load_model()
        self._open_camera()

    def _load_model(self):
        print("Loading realtime DA3 model...")

        expanded_source = os.path.expanduser(
            self.model_source
        )

        if os.path.isdir(expanded_source):
            source = os.path.abspath(expanded_source)
        else:
            source = self.model_source

        self.model = DepthAnything3.from_pretrained(
            source
        )

        self.model = self.model.to(
            device=self.device
        )

        self.model.eval()

        print("Realtime model loaded.")

    def _open_camera(self):
        print(
            f"Opening camera source: "
            f"{self.camera_source}"
        )

        self.cap = cv2.VideoCapture(
            self.camera_source
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara: "
                f"{self.camera_source}"
            )

        # -------------------------------------------------
        # IMPORTANTE:
        # Configurar MJPEG antes de resolución y FPS.
        # -------------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *self.capture_fourcc
        )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            fourcc,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.capture_width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.capture_height,
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            self.capture_fps,
        )

        # Intentar minimizar buffering.
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        actual_width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        actual_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        actual_fourcc_int = int(
            self.cap.get(
                cv2.CAP_PROP_FOURCC
            )
        )

        actual_fourcc = "".join(
            [
                chr(
                    (actual_fourcc_int >> (8 * i))
                    & 0xFF
                )
                for i in range(4)
            ]
        )

        print(
            "Camera opened:",
            f"{actual_width}x{actual_height}",
            f"@ {actual_fps:.2f} FPS",
        )

        print(
            "Camera FOURCC:",
            actual_fourcc,
        )

        if self.stereo_enabled:
            print(
                "Stereo mode: ENABLED"
            )

            print(
                "DA3 input view:",
                self.stereo_view.upper(),
            )

    @staticmethod
    def _extract_single_depth(prediction):
        depth = np.asarray(
            prediction.depth,
            dtype=np.float32,
        )

        depth = np.squeeze(depth)

        if depth.ndim != 2:
            raise ValueError(
                "Se esperaba depth [H, W], "
                f"pero DA3 produjo {depth.shape}."
            )

        return depth

    def _convert_depth(
        self,
        depth_raw,
        input_hw,
    ):
        """
        Convierte la salida de DA3 según el tipo
        de modelo configurado.
        """

        processed_h, processed_w = depth_raw.shape
        input_h, input_w = input_hw

        if self.depth_mode == "metric_focal":
            fx_processed = (
                self.fx
                * processed_w
                / float(input_w)
            )

            fy_processed = (
                self.fy
                * processed_h
                / float(input_h)
            )

            focal_processed = 0.5 * (
                fx_processed + fy_processed
            )

            depth = (
                depth_raw
                * focal_processed
                / self.metric_focal_reference
            )

            unit = "m"

        elif self.depth_mode == "metric_direct":
            depth = depth_raw
            unit = "m"

        else:
            depth = depth_raw
            unit = "relative"

        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        return depth, unit

    @staticmethod
    def _restore_depth_resolution(
        depth,
        target_hw,
    ):
        target_h, target_w = target_hw

        return cv2.resize(
            depth,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _center_depth(depth):
        h, w = depth.shape

        cx = w // 2
        cy = h // 2

        radius = 5

        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)

        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)

        roi = depth[y1:y2, x1:x2]

        valid = roi[
            np.isfinite(roi) & (roi > 0)
        ]

        if valid.size == 0:
            return np.nan

        return float(np.median(valid))

    @staticmethod
    def _colorize_depth(depth):
        valid = depth[
            np.isfinite(depth) & (depth > 0)
        ]

        if valid.size == 0:
            return np.zeros(
                (*depth.shape, 3),
                dtype=np.uint8,
            )

        lower = float(
            np.percentile(valid, 2)
        )

        upper = float(
            np.percentile(valid, 98)
        )

        if upper <= lower:
            upper = lower + 1e-6

        normalized = (
            depth - lower
        ) / (upper - lower)

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        normalized_uint8 = (
            normalized * 255.0
        ).astype(np.uint8)

        return cv2.applyColorMap(
            normalized_uint8,
            cv2.COLORMAP_TURBO,
        )

    def infer_frame(self, frame_bgr):
        input_h, input_w = frame_bgr.shape[:2]

        # OpenCV entrega BGR.
        # DA3 recibe RGB.
        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        if self.device.type == "cuda":
            with torch.inference_mode():
                prediction = self.model.inference(
                    image=[frame_rgb],
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                )
        else:
            with torch.inference_mode():
                prediction = self.model.inference(
                    image=[frame_rgb],
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        self.frame_counter += 1

        if self.frame_counter > self.benchmark_warmup:
            self.benchmark_times.append(elapsed)

        if len(self.benchmark_times) == self.benchmark_frames:
            times = np.array(self.benchmark_times)

            avg_ms = np.mean(times) * 1000.0
            median_ms = np.median(times) * 1000.0
            min_ms = np.min(times) * 1000.0
            max_ms = np.max(times) * 1000.0
            avg_fps = 1.0 / np.mean(times)

            print()
            print("========== BENCHMARK ==========")
            print(f"Frames: {len(times)}")
            print(f"Average latency: {avg_ms:.3f} ms")
            print(f"Median latency:  {median_ms:.3f} ms")
            print(f"Min latency:     {min_ms:.3f} ms")
            print(f"Max latency:     {max_ms:.3f} ms")
            print(f"Average FPS:     {avg_fps:.2f}")
            print("===============================")
            print()

            self.benchmark_times.clear()

        depth_raw = self._extract_single_depth(
            prediction
        )

        depth_processed, unit = self._convert_depth(
            depth_raw,
            input_hw=(input_h, input_w),
        )

        depth_input = self._restore_depth_resolution(
            depth_processed,
            target_hw=(input_h, input_w),
        )

        center_depth = self._center_depth(
            depth_input
        )

        inference_fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )

        return {
            "depth": depth_input,
            "depth_processed": depth_processed,
            "depth_raw": depth_raw,
            "unit": unit,
            "center_depth": center_depth,
            "latency_ms": elapsed * 1000.0,
            "inference_fps": inference_fps,
        }

    def _show_result(
        self,
        frame_bgr,
        result,
    ):
        depth_bgr = self._colorize_depth(
            result["depth"]
        )

        h, w = frame_bgr.shape[:2]

        center = (
            w // 2,
            h // 2,
        )

        cv2.drawMarker(
            frame_bgr,
            center,
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
        )

        center_depth = result[
            "center_depth"
        ]

        unit = result["unit"]

        if np.isfinite(center_depth):
            if unit == "m":
                depth_text = (
                    f"Center: "
                    f"{center_depth:.3f} m"
                )
            else:
                depth_text = (
                    f"Center: "
                    f"{center_depth:.3f} relative"
                )
        else:
            depth_text = "Center: invalid"

        performance_text = (
            f"Inference: "
            f"{result['latency_ms']:.1f} ms | "
            f"{result['inference_fps']:.1f} FPS"
        )

        cv2.putText(
            frame_bgr,
            depth_text,
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame_bgr,
            performance_text,
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if self.display_scale != 1.0:
            display_width = max(
                1,
                int(w * self.display_scale),
            )

            display_height = max(
                1,
                int(h * self.display_scale),
            )

            frame_bgr = cv2.resize(
                frame_bgr,
                (
                    display_width,
                    display_height,
                ),
            )

            depth_bgr = cv2.resize(
                depth_bgr,
                (
                    display_width,
                    display_height,
                ),
            )

        if self.stereo_enabled:
            rgb_window_name = (
                "DA3 Realtime RGB - "
                f"{self.stereo_view.upper()}"
            )
        else:
            rgb_window_name = (
                "DA3 Realtime RGB"
            )

        cv2.imshow(
            rgb_window_name,
            frame_bgr,
        )

        cv2.imshow(
            "DA3 Realtime Metric Depth",
            depth_bgr,
        )

    def run(self):
        print()
        print("Starting realtime inference...")
        print("Q / ESC: salir")
        print()

        try:
            while True:
                success, stereo_frame_bgr = (
                    self.cap.read()
                )

                if not success:
                    print(
                        "No se pudo obtener "
                        "un frame de la cámara."
                    )
                    break

                (
                    frame_bgr,
                    left_frame,
                    right_frame,
                ) = self._split_stereo_frame(
                    stereo_frame_bgr
                )

                result = self.infer_frame(
                    frame_bgr
                )

                if self.display:
                    self._show_result(
                        frame_bgr.copy(),
                        result,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key in (
                        ord("q"),
                        27,
                    ):
                        break

        except KeyboardInterrupt:
            print("\nRealtime interrupted.")

        finally:
            self.close()

    def close(self):
        if hasattr(self, "cap"):
            self.cap.release()

        cv2.destroyAllWindows()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Realtime pipeline closed.")

    def _split_stereo_frame(
        self,
        stereo_frame_bgr,
    ):
        """
        La cámara entrega:

            LEFT | RIGHT

        dentro de un único frame horizontal.
        """

        if not self.stereo_enabled:
            return (
                stereo_frame_bgr,
                None,
                None,
            )

        h, w = stereo_frame_bgr.shape[:2]

        if w % 2 != 0:
            raise ValueError(
                "El ancho del frame estéreo "
                f"debe ser par. Recibido: {w}"
            )

        half_w = w // 2

        left_frame = stereo_frame_bgr[
            :,
            :half_w,
        ]

        right_frame = stereo_frame_bgr[
            :,
            half_w:,
        ]

        if not self._printed_stereo_shape:
            print()
            print(
                "========== STEREO SPLIT =========="
            )
            print(
                "Full stereo frame:",
                stereo_frame_bgr.shape,
            )
            print(
                "Left frame:",
                left_frame.shape,
            )
            print(
                "Right frame:",
                right_frame.shape,
            )
            print(
                "Selected for DA3:",
                self.stereo_view.upper(),
            )
            print(
                "=================================="
            )
            print()

            self._printed_stereo_shape = True

        if self.stereo_view == "left":
            selected_frame = left_frame
        else:
            selected_frame = right_frame

        return (
            selected_frame,
            left_frame,
            right_frame,
        )

def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        print(f"configuration file copied to: {dst_path}")
        return dst_path
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as exc:
        print(f"Copy Error: {exc}")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DA3 cacao offline / realtime"
    )

    parser.add_argument(
        "--mode",
        choices=["offline", "realtime"],
        default="offline",
        help="Modo de ejecución.",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=False,
        help="Image path para modo offline.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./configs/cacao_da3_streaming_hq.yaml",
        help="YAML configuration path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Output path para modo offline.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # =====================================================
    # REALTIME
    # =====================================================

    if args.mode == "realtime":
        realtime = DA3RealtimeDepth(
            config=config
        )

        realtime.run()

        del realtime

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        sys.exit(0)

    # =====================================================
    # OFFLINE / STREAMING ACTUAL
    # =====================================================

    if args.image_dir is None:
        parser.error(
            "--image_dir es obligatorio "
            "cuando --mode=offline."
        )

    image_dir = args.image_dir

    if args.output_dir is not None:
        save_dir = args.output_dir
    else:
        current_datetime = datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )

        exp_dir = "./exps"

        save_dir = os.path.join(
            exp_dir,
            image_dir.replace("/", "_"),
            current_datetime,
        )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        print(
            f"The exp will be saved under "
            f"dir: {save_dir}"
        )

        copy_file(
            args.config,
            save_dir,
        )

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    da3_streaming = DA3_Streaming(
        image_dir,
        save_dir,
        config,
    )

    da3_streaming.run()
    da3_streaming.close()

    del da3_streaming

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    all_ply_path = os.path.join(
        save_dir,
        "pcd/combined_pcd.ply",
    )

    input_dir = os.path.join(
        save_dir,
        "pcd",
    )

    print("Saving all the point clouds")

    merge_ply_files(
        input_dir,
        all_ply_path,
    )

    print("DA3-Streaming done.")

    sys.exit(0)
