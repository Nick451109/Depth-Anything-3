#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib import colormaps


REGISTERED_PAIRS = {
    ("image", "depth"),
    ("image_processed", "depth_processed"),
}


@dataclass(frozen=True)
class GTBox:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float


def frame_number(path: Path) -> int:
    """Extrae el índice de nombres como frame_000123.npz."""
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(
            f"Nombre de archivo no reconocido: {path.name}"
        ) from exc


def load_gt(gt_path: Path) -> dict[int, list[GTBox]]:
    """
    Carga un gt.txt tipo MOT.

    Solo usa las primeras seis columnas:
        frame, id, x, y, width, height

    Las columnas adicionales, si existen, se ignoran.
    """
    if not gt_path.is_file():
        raise FileNotFoundError(f"No existe GT: {gt_path}")

    annotations: dict[int, list[GTBox]] = {}

    with gt_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]

            if len(parts) < 6:
                raise ValueError(
                    f"{gt_path}:{line_number}: se esperaban al menos "
                    f"6 columnas y se encontraron {len(parts)}."
                )

            try:
                frame = int(round(float(parts[0])))
                track_id = int(round(float(parts[1])))
                x = float(parts[2])
                y = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
            except ValueError as exc:
                raise ValueError(
                    f"{gt_path}:{line_number}: valores MOT inválidos: {line}"
                ) from exc

            if width <= 0 or height <= 0:
                continue

            box = GTBox(
                frame=frame,
                track_id=track_id,
                x=x,
                y=y,
                width=width,
                height=height,
            )

            annotations.setdefault(frame, []).append(box)

    return annotations


def valid_values(
    array: np.ndarray,
    positive_only: bool,
) -> np.ndarray:
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
    """Calcula una escala global de profundidad para toda la secuencia."""
    samples: list[np.ndarray] = []

    for path in files:
        with np.load(path) as data:
            if key not in data:
                raise KeyError(
                    f"{path.name} no contiene la clave '{key}'. "
                    f"Claves disponibles: {data.files}"
                )

            values = valid_values(
                data[key],
                positive_only=positive_only,
            )

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
        raise RuntimeError(
            f"No existen valores válidos para '{key}'"
        )

    combined = np.concatenate(samples)

    minimum = float(np.percentile(combined, lower_percentile))
    maximum = float(np.percentile(combined, upper_percentile))

    if maximum <= minimum:
        maximum = minimum + 1e-6

    return minimum, maximum


def normalize_depth(
    depth: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    normalized = (
        depth.astype(np.float32) - minimum
    ) / (maximum - minimum)

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(normalized, 0.0, 1.0)


def colorize_depth(
    depth: np.ndarray,
    minimum: float,
    maximum: float,
    colormap_name: str,
) -> np.ndarray:
    """Convierte depth a una imagen BGR uint8 para OpenCV."""
    normalized = normalize_depth(
        depth,
        minimum,
        maximum,
    )

    try:
        colormap = colormaps[colormap_name]
    except KeyError as exc:
        raise ValueError(
            f"Colormap no reconocido: {colormap_name}"
        ) from exc

    rgb = colormap(normalized)[..., :3]
    rgb = (rgb * 255).astype(np.uint8)

    depth_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    invalid_mask = ~np.isfinite(depth) | (depth <= 0)
    depth_bgr[invalid_mask] = 0

    return depth_bgr


def prepare_rgb(image: np.ndarray) -> np.ndarray:
    """Convierte RGB de DA3 a BGR para OpenCV sin cambiar resolución."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Formato RGB inesperado: shape={image.shape}"
        )

    if image.dtype != np.uint8:
        image = image.astype(np.float32)

        if image.size > 0 and image.max() <= 1.0:
            image *= 255.0

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )


def id_color(track_id: int) -> tuple[int, int, int]:
    """Genera un color BGR estable para cada ID."""
    hue = int((track_id * 47) % 180)

    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

    return tuple(int(value) for value in bgr)


def map_box(
    box: GTBox,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    """
    Mapea una caja del raster original del GT al raster que se visualiza.

    Esto permite usar el mismo GT tanto con image+depth originales como
    con image_processed+depth_processed.
    """
    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)

    x1 = int(round(box.x * scale_x))
    y1 = int(round(box.y * scale_y))
    x2 = int(round((box.x + box.width) * scale_x))
    y2 = int(round((box.y + box.height) * scale_y))

    x1 = int(np.clip(x1, 0, target_width - 1))
    y1 = int(np.clip(y1, 0, target_height - 1))
    x2 = int(np.clip(x2, 0, target_width - 1))
    y2 = int(np.clip(y2, 0, target_height - 1))

    return x1, y1, x2, y2



def estimate_box_metrics(
    depth: np.ndarray,
    boxes: list[GTBox],
    gt_width: int,
    gt_height: int,
    roi_fraction: float,
    fx: float | None = None,
    fy: float | None = None,
) -> dict[int, dict[str, float | None]]:
    """
    Estima métricas por bounding box.

    Retorna:
        track_id -> {
            "depth_m": float | None,
            "width_cm": float | None,
            "height_cm": float | None,
        }

    - depth_m se calcula siempre que haya valores depth válidos.
    - width_cm y height_cm solo se calculan si fx y fy están disponibles.
    """
    if not (0.05 <= roi_fraction <= 1.0):
        raise ValueError("--depth_roi_fraction debe estar entre 0.05 y 1.0.")

    if (fx is None) != (fy is None):
        raise ValueError("fx y fy deben proporcionarse juntos o ambos ser None.")

    if fx is not None and (fx <= 0 or fy <= 0):
        raise ValueError("fx y fy deben ser mayores que cero.")

    depth_h, depth_w = depth.shape
    result: dict[int, dict[str, float | None]] = {}

    for box in boxes:
        x1, y1, x2, y2 = map_box(
            box=box,
            source_width=gt_width,
            source_height=gt_height,
            target_width=depth_w,
            target_height=depth_h,
        )

        if x2 <= x1 or y2 <= y1:
            result[box.track_id] = {
                "depth_m": None,
                "width_cm": None,
                "height_cm": None,
            }
            continue

        box_w_px = float(x2 - x1)
        box_h_px = float(y2 - y1)

        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)

        roi_w = max(1.0, box_w_px * roi_fraction)
        roi_h = max(1.0, box_h_px * roi_fraction)

        rx1 = int(round(center_x - roi_w / 2.0))
        ry1 = int(round(center_y - roi_h / 2.0))
        rx2 = int(round(center_x + roi_w / 2.0))
        ry2 = int(round(center_y + roi_h / 2.0))

        rx1 = int(np.clip(rx1, 0, depth_w - 1))
        ry1 = int(np.clip(ry1, 0, depth_h - 1))
        rx2 = int(np.clip(rx2, rx1 + 1, depth_w))
        ry2 = int(np.clip(ry2, ry1 + 1, depth_h))

        roi = depth[ry1:ry2, rx1:rx2]
        valid = roi[np.isfinite(roi) & (roi > 0)]

        if valid.size == 0:
            result[box.track_id] = {
                "depth_m": None,
                "width_cm": None,
                "height_cm": None,
            }
            continue

        z_m = float(np.median(valid))

        width_cm = None
        height_cm = None

        if fx is not None and fy is not None:
            width_cm = (box_w_px * z_m / fx) * 100.0
            height_cm = (box_h_px * z_m / fy) * 100.0

        result[box.track_id] = {
            "depth_m": z_m,
            "width_cm": width_cm,
            "height_cm": height_cm,
        }

    return result


def extract_focal_lengths(
    intrinsics: np.ndarray | None,
    manual_fx: float | None,
    manual_fy: float | None,
) -> tuple[float, float] | None:
    """
    Prioridad:
      1. --fx y --fy proporcionados manualmente.
      2. Matriz intrinsics guardada en el NPZ.
    """
    if manual_fx is not None or manual_fy is not None:
        if manual_fx is None or manual_fy is None:
            raise ValueError(
                "Debe proporcionar ambos parámetros: --fx y --fy."
            )

        if manual_fx <= 0 or manual_fy <= 0:
            raise ValueError("--fx y --fy deben ser mayores que cero.")

        return float(manual_fx), float(manual_fy)

    if intrinsics is None:
        return None

    matrix = np.asarray(intrinsics)

    # Acepta [3, 3] o [1, 3, 3].
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]

    if matrix.shape != (3, 3):
        return None

    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])

    if not np.isfinite(fx) or not np.isfinite(fy):
        return None

    if fx <= 0 or fy <= 0:
        return None

    return fx, fy


def draw_gt(
    image: np.ndarray,
    boxes: list[GTBox],
    gt_width: int,
    gt_height: int,
    show_size: bool,
    box_metrics: dict[int, dict[str, float | None]] | None = None,
) -> np.ndarray:
    """Dibuja las anotaciones GT sobre una copia de la imagen."""
    result = image.copy()
    image_h, image_w = result.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map_box(
            box=box,
            source_width=gt_width,
            source_height=gt_height,
            target_width=image_w,
            target_height=image_h,
        )

        color = id_color(box.track_id)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        if show_size:
            metrics = (
                box_metrics.get(box.track_id)
                if box_metrics is not None
                else None
            )

            depth_m = None
            width_cm = None
            height_cm = None

            if metrics is not None:
                depth_m = metrics.get("depth_m")
                width_cm = metrics.get("width_cm")
                height_cm = metrics.get("height_cm")

            parts = [f"ID {box.track_id}"]

            if depth_m is not None:
                parts.append(f"z={depth_m:.2f} m")
            else:
                parts.append("z=N/A")

            if width_cm is not None and height_cm is not None:
                parts.append(f"{width_cm:.1f}x{height_cm:.1f} cm")
            else:
                parts.append("cm N/A")

            label = " | ".join(parts)
        else:
            label = f"ID {box.track_id}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        label_x = x1
        label_y = y1 - 8

        if label_y - text_h - baseline < 0:
            label_y = min(image_h - baseline - 2, y1 + text_h + 12)

        background_tl = (
            max(0, label_x),
            max(0, label_y - text_h - 6),
        )
        background_br = (
            min(image_w - 1, label_x + text_w + 8),
            min(image_h - 1, label_y + baseline + 4),
        )

        cv2.rectangle(
            result,
            background_tl,
            background_br,
            color,
            thickness=-1,
        )

        cv2.putText(
            result,
            label,
            (label_x + 4, label_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return result


def map_point(
    x: int,
    y: int,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int]:
    """Convierte una coordenada entre dos cuadrículas."""
    source_h, source_w = source_shape
    target_h, target_w = target_shape

    if source_shape == target_shape:
        return (
            int(np.clip(x, 0, target_w - 1)),
            int(np.clip(y, 0, target_h - 1)),
        )

    target_x = int(round((x + 0.5) * target_w / source_w - 0.5))
    target_y = int(round((y + 0.5) * target_h / source_h - 0.5))

    return (
        int(np.clip(target_x, 0, target_w - 1)),
        int(np.clip(target_y, 0, target_h - 1)),
    )


def valid_depth_value(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def refresh_measurement(state: dict[str, Any]) -> None:
    depth = state.get("depth")
    rgb_shape = state.get("rgb_shape")
    depth_shape = state.get("depth_shape")
    source = state.get("selection_source")
    source_point = state.get("selection_point")

    if (
        depth is None
        or rgb_shape is None
        or depth_shape is None
        or source is None
        or source_point is None
    ):
        state["measurement"] = None
        return

    source_x, source_y = source_point

    if source == "rgb":
        rgb_x = int(np.clip(source_x, 0, rgb_shape[1] - 1))
        rgb_y = int(np.clip(source_y, 0, rgb_shape[0] - 1))

        depth_x, depth_y = map_point(
            rgb_x,
            rgb_y,
            rgb_shape,
            depth_shape,
        )

    elif source == "depth":
        depth_x = int(np.clip(source_x, 0, depth_shape[1] - 1))
        depth_y = int(np.clip(source_y, 0, depth_shape[0] - 1))

        rgb_x, rgb_y = map_point(
            depth_x,
            depth_y,
            depth_shape,
            rgb_shape,
        )

    else:
        raise ValueError(f"Fuente de selección inválida: {source}")

    depth_value = float(depth[depth_y, depth_x])

    confidence_value: float | None = None
    confidence = state.get("confidence")

    if (
        isinstance(confidence, np.ndarray)
        and confidence.ndim == 2
        and confidence.shape == depth.shape
    ):
        confidence_value = float(confidence[depth_y, depth_x])

    state["measurement"] = {
        "rgb_point": (rgb_x, rgb_y),
        "depth_point": (depth_x, depth_y),
        "depth_value": depth_value,
        "confidence_value": confidence_value,
    }


def print_measurement(state: dict[str, Any]) -> None:
    measurement = state.get("measurement")

    if measurement is None:
        return

    frame_id = state["frame_id"]
    gt_frame_id = state["gt_frame_id"]

    rgb_x, rgb_y = measurement["rgb_point"]
    depth_x, depth_y = measurement["depth_point"]
    depth_value = measurement["depth_value"]
    confidence_value = measurement["confidence_value"]
    unit = state["depth_unit"]

    if valid_depth_value(depth_value):
        depth_text = f"{depth_value:.3f} {unit}"
    else:
        depth_text = "inválida"

    message = (
        f"NPZ {frame_id:06d} | GT frame {gt_frame_id} | "
        f"RGB(x={rgb_x}, y={rgb_y}) | "
        f"Depth(x={depth_x}, y={depth_y}) | "
        f"profundidad={depth_text}"
    )

    if confidence_value is not None and np.isfinite(confidence_value):
        message += f" | conf={confidence_value:.3f}"

    print(f"\n{message}")


def display_to_image_point(
    x: int,
    y: int,
    display_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    return map_point(
        x=x,
        y=y,
        source_shape=display_shape,
        target_shape=image_shape,
    )


def make_mouse_callback(
    source: str,
    state: dict[str, Any],
):
    """
    Clic izquierdo: selecciona un punto.
    Clic derecho: elimina la selección.

    Si la ventana está reducida con --display_scale, convierte el punto
    de la ventana a la coordenada real del raster antes de medir depth.
    """
    def callback(
        event: int,
        x: int,
        y: int,
        flags: int,
        param: Any,
    ) -> None:
        del flags, param

        if event == cv2.EVENT_LBUTTONDOWN:
            if source == "rgb":
                image_shape = state.get("rgb_shape")
                display_shape = state.get("rgb_display_shape")
            else:
                image_shape = state.get("depth_shape")
                display_shape = state.get("depth_display_shape")

            if image_shape is None or display_shape is None:
                return

            image_x, image_y = display_to_image_point(
                x=x,
                y=y,
                display_shape=display_shape,
                image_shape=image_shape,
            )

            state["selection_source"] = source
            state["selection_point"] = (image_x, image_y)

            refresh_measurement(state)
            print_measurement(state)

        elif event == cv2.EVENT_RBUTTONDOWN:
            state["selection_source"] = None
            state["selection_point"] = None
            state["measurement"] = None
            print("\nSelección eliminada.")

    return callback


def draw_measurement(
    image: np.ndarray,
    point: tuple[int, int] | None,
    measurement: dict[str, Any] | None,
    depth_unit: str,
) -> np.ndarray:
    """Dibuja la medición en una copia sin modificar el raster."""
    result = image.copy()

    if point is None or measurement is None:
        return result

    x, y = point
    height, width = result.shape[:2]

    if not (0 <= x < width and 0 <= y < height):
        return result

    depth_value = measurement["depth_value"]
    confidence_value = measurement["confidence_value"]

    if valid_depth_value(depth_value):
        depth_text = f"{depth_value:.3f} {depth_unit}"
    else:
        depth_text = "profundidad inválida"

    label = f"({x}, {y})  {depth_text}"

    if confidence_value is not None and np.isfinite(confidence_value):
        label += f"  conf={confidence_value:.2f}"

    cv2.drawMarker(
        result,
        (x, y),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=26,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    label_x = x + 14
    label_y = y - 14

    if label_x + text_w + 12 >= width:
        label_x = max(5, x - text_w - 20)

    if label_y - text_h - 10 < 0:
        label_y = min(height - baseline - 8, y + text_h + 22)

    top_left = (
        max(0, label_x - 6),
        max(0, label_y - text_h - 8),
    )
    bottom_right = (
        min(width - 1, label_x + text_w + 6),
        min(height - 1, label_y + baseline + 6),
    )

    cv2.rectangle(
        result,
        top_left,
        bottom_right,
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        result,
        label,
        (label_x, label_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return result


def resize_for_display(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Reduce solo la copia mostrada.

    Los arrays RGB/depth originales y las coordenadas GT no se modifican.
    """
    if scale == 1.0:
        return image

    height, width = image.shape[:2]

    display_width = max(1, int(round(width * scale)))
    display_height = max(1, int(round(height * scale)))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        image,
        (display_width, display_height),
        interpolation=interpolation,
    )


def load_frame(
    path: Path,
    rgb_key: str,
    depth_key: str,
    confidence_key: str,
    intrinsics_key: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    with np.load(path) as data:
        if rgb_key not in data:
            raise KeyError(
                f"{path.name} no contiene '{rgb_key}'. "
                f"Claves disponibles: {data.files}"
            )

        if depth_key not in data:
            raise KeyError(
                f"{path.name} no contiene '{depth_key}'. "
                f"Claves disponibles: {data.files}"
            )

        rgb = np.asarray(data[rgb_key]).copy()
        depth = np.asarray(data[depth_key]).copy()

        confidence: np.ndarray | None = None
        intrinsics: np.ndarray | None = None

        if confidence_key in data:
            confidence = np.asarray(data[confidence_key]).copy()

        if intrinsics_key in data:
            intrinsics = np.asarray(data[intrinsics_key]).copy()

    if depth.ndim != 2:
        raise ValueError(
            f"Depth debe ser [H, W], pero se obtuvo "
            f"{depth.shape} en {path.name}"
        )

    if confidence is not None and confidence.ndim != 2:
        print(
            f"ADVERTENCIA: {confidence_key} tiene shape "
            f"{confidence.shape}; se ignorará."
        )
        confidence = None

    return rgb, depth, confidence, intrinsics


def validate_pair(
    rgb_key: str,
    depth_key: str,
    rgb_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> None:
    pair = (rgb_key, depth_key)

    if pair not in REGISTERED_PAIRS:
        print(
            "ADVERTENCIA: pareja RGB/depth no registrada. "
            "Las coordenadas se mapearán proporcionalmente."
        )

    if pair in REGISTERED_PAIRS and rgb_shape != depth_shape:
        raise ValueError(
            f"La pareja registrada {rgb_key} + {depth_key} debería tener "
            f"la misma resolución, pero se obtuvo RGB={rgb_shape[::-1]} "
            f"y depth={depth_shape[::-1]}."
        )


def update_window_title(
    window_name: str,
    content_name: str,
    npz_frame: int,
    gt_frame: int,
    boxes: int,
) -> None:
    title = (
        f"{content_name} | NPZ {npz_frame:06d} | "
        f"GT {gt_frame} | objetos {boxes}"
    )

    try:
        cv2.setWindowTitle(window_name, title)
    except cv2.error:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualiza RGB y depth de DA3 con las anotaciones MOT "
            "del ground truth sobre ambas vistas y muestra profundidad + tamaño estimado en cm."
        )
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directorio principal de salida de DA3-Streaming.",
    )

    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(
            "/home/cidis/Documents/GitHub/Depth-Anything-3/"
            "sequences/real/seq001/gt/gt.txt"
        ),
        help="Ruta al gt.txt tipo MOT.",
    )

    parser.add_argument(
        "--gt_width",
        type=int,
        default=1920,
        help="Ancho del raster sobre el que fue anotado el GT.",
    )

    parser.add_argument(
        "--gt_height",
        type=int,
        default=1080,
        help="Alto del raster sobre el que fue anotado el GT.",
    )

    parser.add_argument(
        "--gt_frame_offset",
        type=int,
        default=1,
        help=(
            "GT_frame = indice_NPZ + offset. "
            "Use 1 para MOT 1-based con frame_000000.npz."
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Velocidad de reproducción.",
    )

    parser.add_argument(
        "--display_scale",
        type=float,
        default=0.5,
        help=(
            "Escala solo de visualización. "
            "0.5 muestra 1920x1080 como 960x540. "
            "Use 1.0 para tamaño original."
        ),
    )

    parser.add_argument(
        "--rgb_key",
        type=str,
        default="image",
        help="Clave RGB dentro del NPZ.",
    )

    parser.add_argument(
        "--depth_key",
        type=str,
        default="depth",
        help="Clave depth dentro del NPZ.",
    )

    parser.add_argument(
        "--confidence_key",
        type=str,
        default="conf",
        help="Clave confidence dentro del NPZ.",
    )

    parser.add_argument(
        "--intrinsics_key",
        type=str,
        default="intrinsics",
        help="Clave de la matriz intrínseca dentro del NPZ.",
    )

    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help=(
            "Focal horizontal en píxeles. Si se especifica, "
            "también debe especificarse --fy. Tiene prioridad "
            "sobre intrinsics del NPZ."
        ),
    )

    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help=(
            "Focal vertical en píxeles. Si se especifica, "
            "también debe especificarse --fx."
        ),
    )

    parser.add_argument(
        "--depth_roi_fraction",
        type=float,
        default=0.50,
        help=(
            "Fracción central de la bounding box usada para calcular "
            "la profundidad mediana del cacao. Default: 0.50."
        ),
    )

    parser.add_argument(
        "--depth_unit",
        type=str,
        default="m",
        help="Unidad mostrada para depth. No convierte valores.",
    )

    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo_r",
        help="Colormap Matplotlib para depth.",
    )

    parser.add_argument(
        "--hide_size",
        action="store_true",
        help="Muestra solo el ID y oculta width x height del GT.",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Reinicia al llegar al último frame.",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps debe ser mayor que cero.")

    if args.display_scale <= 0:
        raise ValueError("--display_scale debe ser mayor que cero.")

    if args.gt_width <= 0 or args.gt_height <= 0:
        raise ValueError("--gt_width y --gt_height deben ser positivos.")

    if (args.fx is None) != (args.fy is None):
        raise ValueError("Debe proporcionar --fx y --fy juntos.")

    if not (0.05 <= args.depth_roi_fraction <= 1.0):
        raise ValueError(
            "--depth_roi_fraction debe estar entre 0.05 y 1.0."
        )

    results_dir = args.output_dir / "results_output"

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
            f"No existen frame_*.npz en {results_dir}"
        )

    gt_by_frame = load_gt(args.gt)

    total_gt_boxes = sum(
        len(boxes)
        for boxes in gt_by_frame.values()
    )

    print(f"Frames NPZ encontrados: {len(files)}")
    print(f"GT:                    {args.gt}")
    print(f"Frames con GT:         {len(gt_by_frame)}")
    print(f"Anotaciones GT:        {total_gt_boxes}")
    print(
        f"Correspondencia: GT_frame = NPZ_frame + "
        f"{args.gt_frame_offset}"
    )
    print(
        f"Raster GT:             "
        f"{args.gt_width}x{args.gt_height}"
    )
    print(
        f"Escala visual:         "
        f"{args.display_scale:.2f}"
    )
    print("Calculando escala global de profundidad...")

    depth_limits = estimate_limits(
        files=files,
        key=args.depth_key,
        lower_percentile=2,
        upper_percentile=98,
        positive_only=True,
    )

    print(f"Límites globales depth: {depth_limits}")

    rgb_window = "DA3 RGB + GT"
    depth_window = "DA3 Depth + GT"

    cv2.namedWindow(rgb_window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(depth_window, cv2.WINDOW_AUTOSIZE)

    state: dict[str, Any] = {
        "frame_id": 0,
        "gt_frame_id": 0,
        "depth": None,
        "confidence": None,
        "rgb_shape": None,
        "depth_shape": None,
        "rgb_display_shape": None,
        "depth_display_shape": None,
        "selection_source": None,
        "selection_point": None,
        "measurement": None,
        "depth_unit": args.depth_unit,
    }

    cv2.setMouseCallback(
        rgb_window,
        make_mouse_callback("rgb", state),
    )

    cv2.setMouseCallback(
        depth_window,
        make_mouse_callback("depth", state),
    )

    delay_ms = max(
        1,
        int(round(1000.0 / args.fps)),
    )

    paused = False
    index = 0
    loaded_index: int | None = None

    rgb_bgr: np.ndarray | None = None
    depth_bgr: np.ndarray | None = None
    current_boxes: list[GTBox] = []
    current_box_metrics: dict[
        int,
        dict[str, float | None],
    ] = {}
    warned_missing_intrinsics = False

    print()
    print("Controles:")
    print("  Clic izquierdo: medir profundidad")
    print("  Clic derecho: eliminar medición")
    print("  ESPACIO: pausar / continuar")
    print("  N: siguiente frame")
    print("  P: frame anterior")
    print("  R: eliminar medición")
    print("  Q o ESC: cerrar")
    print()

    try:
        while True:
            if loaded_index != index:
                path = files[index]

                npz_frame = frame_number(path)
                gt_frame = npz_frame + args.gt_frame_offset

                rgb, depth, confidence, intrinsics = load_frame(
                    path=path,
                    rgb_key=args.rgb_key,
                    depth_key=args.depth_key,
                    confidence_key=args.confidence_key,
                    intrinsics_key=args.intrinsics_key,
                )

                rgb_bgr = prepare_rgb(rgb)

                depth_bgr = colorize_depth(
                    depth=depth,
                    minimum=depth_limits[0],
                    maximum=depth_limits[1],
                    colormap_name=args.colormap,
                )

                rgb_shape = rgb_bgr.shape[:2]
                depth_shape = depth_bgr.shape[:2]

                validate_pair(
                    rgb_key=args.rgb_key,
                    depth_key=args.depth_key,
                    rgb_shape=rgb_shape,
                    depth_shape=depth_shape,
                )

                current_boxes = gt_by_frame.get(
                    gt_frame,
                    [],
                )

                focal_lengths = extract_focal_lengths(
                    intrinsics=intrinsics,
                    manual_fx=args.fx,
                    manual_fy=args.fy,
                )

                if focal_lengths is None:
                    current_box_metrics = estimate_box_metrics(
                        depth=depth,
                        boxes=current_boxes,
                        gt_width=args.gt_width,
                        gt_height=args.gt_height,
                        roi_fraction=args.depth_roi_fraction,
                        fx=None,
                        fy=None,
                    )

                    if not warned_missing_intrinsics:
                        print()
                        print(
                            "ADVERTENCIA: no se encontraron intrinsics en "
                            "el NPZ y no se proporcionaron --fx/--fy. "
                            "La profundidad se mostrará igual, pero el "
                            "tamaño físico se mostrará como cm N/A."
                        )
                        print(
                            "Revise las claves del NPZ o ejecute el script "
                            "con --fx <valor> --fy <valor>."
                        )
                        warned_missing_intrinsics = True
                else:
                    fx, fy = focal_lengths

                    current_box_metrics = estimate_box_metrics(
                        depth=depth,
                        boxes=current_boxes,
                        gt_width=args.gt_width,
                        gt_height=args.gt_height,
                        roi_fraction=args.depth_roi_fraction,
                        fx=fx,
                        fy=fy,
                    )

                state["frame_id"] = npz_frame
                state["gt_frame_id"] = gt_frame
                state["depth"] = depth
                state["confidence"] = confidence
                state["rgb_shape"] = rgb_shape
                state["depth_shape"] = depth_shape

                refresh_measurement(state)

                update_window_title(
                    window_name=rgb_window,
                    content_name=args.rgb_key,
                    npz_frame=npz_frame,
                    gt_frame=gt_frame,
                    boxes=len(current_boxes),
                )

                update_window_title(
                    window_name=depth_window,
                    content_name=args.depth_key,
                    npz_frame=npz_frame,
                    gt_frame=gt_frame,
                    boxes=len(current_boxes),
                )

                loaded_index = index

            if rgb_bgr is None or depth_bgr is None:
                raise RuntimeError("No se pudo cargar el frame actual.")

            rgb_annotated = draw_gt(
                image=rgb_bgr,
                boxes=current_boxes,
                gt_width=args.gt_width,
                gt_height=args.gt_height,
                show_size=not args.hide_size,
                box_metrics=current_box_metrics,
            )

            depth_annotated = draw_gt(
                image=depth_bgr,
                boxes=current_boxes,
                gt_width=args.gt_width,
                gt_height=args.gt_height,
                show_size=not args.hide_size,
                box_metrics=current_box_metrics,
            )

            measurement = state.get("measurement")
            rgb_point = None
            depth_point = None

            if measurement is not None:
                rgb_point = measurement["rgb_point"]
                depth_point = measurement["depth_point"]

            rgb_annotated = draw_measurement(
                image=rgb_annotated,
                point=rgb_point,
                measurement=measurement,
                depth_unit=args.depth_unit,
            )

            depth_annotated = draw_measurement(
                image=depth_annotated,
                point=depth_point,
                measurement=measurement,
                depth_unit=args.depth_unit,
            )

            rgb_display = resize_for_display(
                rgb_annotated,
                args.display_scale,
            )

            depth_display = resize_for_display(
                depth_annotated,
                args.display_scale,
            )

            state["rgb_display_shape"] = rgb_display.shape[:2]
            state["depth_display_shape"] = depth_display.shape[:2]

            cv2.imshow(rgb_window, rgb_display)
            cv2.imshow(depth_window, depth_display)

            print(
                f"\rNPZ {state['frame_id']:06d} | "
                f"GT {state['gt_frame_id']} | "
                f"objetos={len(current_boxes)} | "
                f"{'PAUSA' if paused else 'REPRODUCIENDO'}",
                end="",
                flush=True,
            )

            wait_ms = 30 if paused else delay_ms
            key = cv2.waitKeyEx(wait_ms)

            if key in (27, ord("q"), ord("Q")):
                break

            if key == ord(" "):
                paused = not paused
                continue

            if key in (ord("r"), ord("R")):
                state["selection_source"] = None
                state["selection_point"] = None
                state["measurement"] = None
                continue

            if key in (ord("n"), ord("N")):
                index = min(index + 1, len(files) - 1)
                paused = True
                continue

            if key in (ord("p"), ord("P")):
                index = max(index - 1, 0)
                paused = True
                continue

            if not paused:
                index += 1

                if index >= len(files):
                    if args.loop:
                        index = 0
                    else:
                        break

    finally:
        cv2.destroyAllWindows()

    print()
    print("Visualización terminada.")


if __name__ == "__main__":
    main()