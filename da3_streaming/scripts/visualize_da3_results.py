#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps


def frame_number(path: Path) -> int:
    """Extrae el índice de frame_123.npz."""
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(
            f"Nombre de archivo no reconocido: {path.name}"
        ) from exc


def valid_values(array: np.ndarray, positive_only: bool) -> np.ndarray:
    mask = np.isfinite(array)

    if positive_only:
        mask &= array > 0

    return array[mask]


def estimate_limits(
    files: list[Path],
    key: str,
    lower_percentile: float,
    upper_percentile: float,
    positive_only: bool = False,
    samples_per_frame: int = 5000,
) -> tuple[float, float]:
    """
    Calcula una escala global para toda la secuencia.

    Esto evita que cada frame use colores diferentes para una misma
    profundidad.
    """
    samples: list[np.ndarray] = []

    for path in files:
        with np.load(path) as data:
            if key not in data:
                raise KeyError(f"{path.name} no contiene '{key}'")

            values = valid_values(data[key], positive_only)

            if values.size == 0:
                continue

            if values.size > samples_per_frame:
                indices = np.linspace(
                    0,
                    values.size - 1,
                    samples_per_frame,
                    dtype=np.int64,
                )
                values = values[indices]

            samples.append(values.astype(np.float32))

    if not samples:
        raise RuntimeError(f"No existen valores válidos para '{key}'")

    combined = np.concatenate(samples)

    minimum = float(np.percentile(combined, lower_percentile))
    maximum = float(np.percentile(combined, upper_percentile))

    if maximum <= minimum:
        maximum = minimum + 1e-6

    return minimum, maximum


def normalize(
    array: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    normalized = (array.astype(np.float32) - minimum) / (
        maximum - minimum
    )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(normalized, 0.0, 1.0)


def colorize(
    normalized: np.ndarray,
    colormap_name: str,
) -> np.ndarray:
    """
    Devuelve una imagen BGR uint8 para OpenCV.
    """
    colormap = colormaps[colormap_name]
    rgb = colormap(normalized)[..., :3]
    rgb = (rgb * 255).astype(np.uint8)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def prepare_rgb(image: np.ndarray) -> np.ndarray:
    """
    DA3 guarda image como [H, W, 3] uint8 RGB.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Formato RGB inesperado: shape={image.shape}"
        )

    if image.dtype != np.uint8:
        image = image.astype(np.float32)

        if image.max() <= 1.0:
            image *= 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()

    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 42),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        result,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def build_panel(
    rgb: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    depth_limits: tuple[float, float],
    confidence_limits: tuple[float, float],
    frame_id: int,
) -> np.ndarray:
    rgb_bgr = prepare_rgb(rgb)

    depth_normalized = normalize(
        depth,
        depth_limits[0],
        depth_limits[1],
    )

    confidence_normalized = normalize(
        confidence,
        confidence_limits[0],
        confidence_limits[1],
    )

    # turbo_r: profundidades menores aparecen cálidas.
    depth_color = colorize(depth_normalized, "turbo_r")
    confidence_color = colorize(confidence_normalized, "viridis")

    if depth_color.shape[:2] != rgb_bgr.shape[:2]:
        depth_color = cv2.resize(
            depth_color,
            (rgb_bgr.shape[1], rgb_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if confidence_color.shape[:2] != rgb_bgr.shape[:2]:
        confidence_color = cv2.resize(
            confidence_color,
            (rgb_bgr.shape[1], rgb_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = cv2.addWeighted(
        rgb_bgr,
        0.55,
        depth_color,
        0.45,
        0.0,
    )

    rgb_bgr = add_label(rgb_bgr, f"RGB — frame {frame_id}")
    depth_color = add_label(
        depth_color,
        "Depth — cálido: cerca | frío: lejos",
    )
    confidence_color = add_label(
        confidence_color,
        "Confianza — claro: mayor confianza",
    )
    overlay = add_label(overlay, "RGB + Depth")

    top = cv2.hconcat([rgb_bgr, depth_color])
    bottom = cv2.hconcat([confidence_color, overlay])

    return cv2.vconcat([top, bottom])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una visualización RGB, depth y confidence "
            "para los resultados de DA3-Streaming."
        )
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directorio principal de salida de DA3-Streaming.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="FPS del video de salida.",
    )

    args = parser.parse_args()

    results_dir = args.output_dir / "results_output"
    visualization_dir = args.output_dir / "visualization"

    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"No existe results_output: {results_dir}"
        )

    files = sorted(
        results_dir.glob("frame_*.npz"),
        key=frame_number,
    )

    if not files:
        raise RuntimeError(
            f"No existen archivos frame_*.npz en {results_dir}"
        )

    visualization_dir.mkdir(parents=True, exist_ok=True)

    print(f"Frames encontrados: {len(files)}")
    print("Calculando escala global de profundidad...")

    depth_limits = estimate_limits(
        files,
        key="depth",
        lower_percentile=2,
        upper_percentile=98,
        positive_only=True,
    )

    confidence_limits = estimate_limits(
        files,
        key="conf",
        lower_percentile=1,
        upper_percentile=99,
        positive_only=False,
    )

    print("Depth global:", depth_limits)
    print("Confidence global:", confidence_limits)

    with np.load(files[0]) as first_data:
        first_panel = build_panel(
            rgb=first_data["image"],
            depth=first_data["depth"],
            confidence=first_data["conf"],
            depth_limits=depth_limits,
            confidence_limits=confidence_limits,
            frame_id=frame_number(files[0]),
        )

    video_path = visualization_dir / "seq007_da3_preview.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (first_panel.shape[1], first_panel.shape[0]),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "No se pudo crear el video MP4 con OpenCV."
        )

    sample_indices = {
        0,
        len(files) // 2,
        len(files) - 1,
    }

    try:
        for index, path in enumerate(files):
            with np.load(path) as data:
                panel = build_panel(
                    rgb=data["image"],
                    depth=data["depth"],
                    confidence=data["conf"],
                    depth_limits=depth_limits,
                    confidence_limits=confidence_limits,
                    frame_id=frame_number(path),
                )

            writer.write(panel)

            if index in sample_indices:
                sample_path = (
                    visualization_dir
                    / f"preview_{frame_number(path):06d}.png"
                )
                cv2.imwrite(str(sample_path), panel)

            print(
                f"\rProcesando {index + 1}/{len(files)}",
                end="",
                flush=True,
            )
    finally:
        writer.release()

    print()
    print(f"Video: {video_path}")
    print(f"Muestras: {visualization_dir}")


if __name__ == "__main__":
    main()