#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib import colormaps


REGISTERED_PAIRS = {
    ("image", "depth"),
    ("image_processed", "depth_processed"),
}


def frame_number(path: Path) -> int:
    """Extrae el índice de nombres como frame_000123.npz."""
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(
            f"Nombre de archivo no reconocido: {path.name}"
        ) from exc


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
    """
    Calcula una escala global para toda la secuencia.

    Así, un mismo valor de profundidad mantiene aproximadamente
    el mismo color entre frames.
    """
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

    minimum = float(
        np.percentile(combined, lower_percentile)
    )
    maximum = float(
        np.percentile(combined, upper_percentile)
    )

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
    """
    Convierte depth a una imagen BGR uint8 para OpenCV.

    No cambia el alto ni el ancho del mapa de profundidad.
    """
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
    """
    Convierte una imagen RGB almacenada por DA3 a BGR para OpenCV.

    No cambia la resolución de la imagen.
    """
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


def update_window_title(
    window_name: str,
    content_name: str,
    frame_id: int,
    image: np.ndarray,
) -> None:
    height, width = image.shape[:2]

    title = (
        f"{content_name} | frame {frame_id:06d} | "
        f"{width}x{height}"
    )

    try:
        cv2.setWindowTitle(window_name, title)
    except cv2.error:
        pass


def map_point(
    x: int,
    y: int,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int]:
    """
    Convierte una coordenada entre dos cuadrículas sin modificar imágenes.

    Cuando ambas resoluciones son iguales, la correspondencia es directa.
    Si son diferentes, usa una transformación proporcional únicamente
    para permitir la inspección visual entre ventanas independientes.
    """
    source_h, source_w = source_shape
    target_h, target_w = target_shape

    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"Resolución fuente inválida: {source_shape}")

    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Resolución destino inválida: {target_shape}")

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
    """Recalcula profundidad y confianza para el punto seleccionado."""
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
        f"Frame {frame_id:06d} | "
        f"RGB(x={rgb_x}, y={rgb_y}) | "
        f"Depth(x={depth_x}, y={depth_y}) | "
        f"profundidad={depth_text}"
    )

    if confidence_value is not None and np.isfinite(confidence_value):
        message += f" | conf={confidence_value:.3f}"

    print(f"\n{message}")


def make_mouse_callback(
    source: str,
    state: dict[str, Any],
):
    """
    Clic izquierdo: selecciona un punto.
    Clic derecho: elimina la selección.
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
            state["selection_source"] = source
            state["selection_point"] = (x, y)
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
    """Dibuja la medición en una copia, sin cambiar la resolución."""
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


def load_frame(
    path: Path,
    rgb_key: str,
    depth_key: str,
    confidence_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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
        if confidence_key in data:
            confidence = np.asarray(data[confidence_key]).copy()

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

    return rgb, depth, confidence


def validate_pair(
    rgb_key: str,
    depth_key: str,
    rgb_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> None:
    pair = (rgb_key, depth_key)

    if pair not in REGISTERED_PAIRS:
        print(
            "ADVERTENCIA: la pareja seleccionada no es una de las "
            "parejas registradas recomendadas:"
        )
        print("  image + depth")
        print("  image_processed + depth_processed")
        print(
            "La coordenada entre ventanas se mapeará proporcionalmente, "
            "pero eso no garantiza correspondencia geométrica exacta."
        )

    if pair in REGISTERED_PAIRS and rgb_shape != depth_shape:
        raise ValueError(
            f"La pareja registrada {rgb_key} + {depth_key} debería tener "
            f"la misma resolución, pero se obtuvo RGB={rgb_shape[::-1]} "
            f"y depth={depth_shape[::-1]}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualiza RGB y profundidad de DA3 en dos ventanas "
            "independientes, conserva sus resoluciones y permite "
            "medir la profundidad de un píxel con el mouse."
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
        help="Velocidad de reproducción.",
    )

    parser.add_argument(
        "--rgb_key",
        type=str,
        default="image",
        help=(
            "Clave RGB dentro del NPZ. "
            "Use image o image_processed."
        ),
    )

    parser.add_argument(
        "--depth_key",
        type=str,
        default="depth",
        help=(
            "Clave de profundidad dentro del NPZ. "
            "Use depth o depth_processed."
        ),
    )

    parser.add_argument(
        "--confidence_key",
        type=str,
        default="conf",
        help=(
            "Clave de confianza dentro del NPZ. "
            "Para depth_processed normalmente use conf_processed."
        ),
    )

    parser.add_argument(
        "--depth_unit",
        type=str,
        default="m",
        help=(
            "Texto de unidad mostrado junto a la profundidad. "
            "No realiza conversiones numéricas."
        ),
    )

    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo_r",
        help="Mapa de color de Matplotlib para profundidad.",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Reinicia la reproducción al llegar al último frame.",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps debe ser mayor que cero.")

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
            f"No existen archivos frame_*.npz en {results_dir}"
        )

    print(f"Frames encontrados: {len(files)}")
    print(f"RGB seleccionado:   {args.rgb_key}")
    print(f"Depth seleccionado: {args.depth_key}")
    print(f"Confidence:          {args.confidence_key}")
    print("Calculando escala global de profundidad...")

    depth_limits = estimate_limits(
        files=files,
        key=args.depth_key,
        lower_percentile=2,
        upper_percentile=98,
        positive_only=True,
    )

    print("Límites globales de depth:", depth_limits)

    rgb_window = "DA3 RGB"
    depth_window = "DA3 Depth"

    cv2.namedWindow(
        rgb_window,
        cv2.WINDOW_AUTOSIZE,
    )
    cv2.namedWindow(
        depth_window,
        cv2.WINDOW_AUTOSIZE,
    )

    state: dict[str, Any] = {
        "frame_id": 0,
        "depth": None,
        "confidence": None,
        "rgb_shape": None,
        "depth_shape": None,
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

    print()
    print("Controles:")
    print("  Clic izquierdo: seleccionar punto y medir profundidad")
    print("  Clic derecho: eliminar punto seleccionado")
    print("  ESPACIO: pausar o continuar")
    print("  N: siguiente frame")
    print("  P: frame anterior")
    print("  R: eliminar punto seleccionado")
    print("  Q o ESC: cerrar")
    print()

    try:
        while True:
            if loaded_index != index:
                path = files[index]
                current_frame = frame_number(path)

                rgb, depth, confidence = load_frame(
                    path=path,
                    rgb_key=args.rgb_key,
                    depth_key=args.depth_key,
                    confidence_key=args.confidence_key,
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

                state["frame_id"] = current_frame
                state["depth"] = depth
                state["confidence"] = confidence
                state["rgb_shape"] = rgb_shape
                state["depth_shape"] = depth_shape

                refresh_measurement(state)

                update_window_title(
                    window_name=rgb_window,
                    content_name=args.rgb_key,
                    frame_id=current_frame,
                    image=rgb_bgr,
                )
                update_window_title(
                    window_name=depth_window,
                    content_name=args.depth_key,
                    frame_id=current_frame,
                    image=depth_bgr,
                )

                loaded_index = index

            if rgb_bgr is None or depth_bgr is None:
                raise RuntimeError("No se pudo cargar el frame actual.")

            measurement = state.get("measurement")
            rgb_point = None
            depth_point = None

            if measurement is not None:
                rgb_point = measurement["rgb_point"]
                depth_point = measurement["depth_point"]

            rgb_display = draw_measurement(
                image=rgb_bgr,
                point=rgb_point,
                measurement=measurement,
                depth_unit=args.depth_unit,
            )
            depth_display = draw_measurement(
                image=depth_bgr,
                point=depth_point,
                measurement=measurement,
                depth_unit=args.depth_unit,
            )

            cv2.imshow(rgb_window, rgb_display)
            cv2.imshow(depth_window, depth_display)

            rgb_h, rgb_w = rgb_bgr.shape[:2]
            depth_h, depth_w = depth_bgr.shape[:2]

            print(
                f"\rFrame {state['frame_id']:06d} | "
                f"RGB: {rgb_w}x{rgb_h} | "
                f"Depth: {depth_w}x{depth_h} | "
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