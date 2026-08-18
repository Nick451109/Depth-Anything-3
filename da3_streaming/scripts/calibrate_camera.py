#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import yaml


# ============================================================
# ARGUMENTOS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibración intrínseca de cámara mediante "
            "checkerboard. Soporta cámaras monoculares y "
            "estéreo side-by-side."
        )
    )

    # --------------------------------------------------------
    # Cámara
    # --------------------------------------------------------

    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help=(
            "Índice o dispositivo de cámara. "
            "Ejemplos: 0, 1, /dev/video0."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Ancho TOTAL solicitado a la cámara.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Alto TOTAL solicitado a la cámara.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS solicitados.",
    )

    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help=(
            "Formato FOURCC. "
            "Ejemplos: MJPG, YUYV."
        ),
    )

    # --------------------------------------------------------
    # Tipo de imagen
    # --------------------------------------------------------

    parser.add_argument(
        "--view",
        choices=[
            "mono",
            "left",
            "right",
        ],
        default="mono",
        help=(
            "mono: usar frame completo. "
            "left/right: asumir cámara estéreo "
            "side-by-side y usar esa mitad."
        ),
    )

    # --------------------------------------------------------
    # Checkerboard
    # --------------------------------------------------------

    parser.add_argument(
        "--board-cols",
        type=int,
        required=True,
        help=(
            "Número de ESQUINAS INTERNAS "
            "horizontales."
        ),
    )

    parser.add_argument(
        "--board-rows",
        type=int,
        required=True,
        help=(
            "Número de ESQUINAS INTERNAS "
            "verticales."
        ),
    )

    parser.add_argument(
        "--square-mm",
        type=float,
        required=True,
        help=(
            "Tamaño físico de cada cuadrado "
            "del checkerboard en milímetros."
        ),
    )

    # --------------------------------------------------------
    # Capturas
    # --------------------------------------------------------

    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help=(
            "Número mínimo de imágenes antes "
            "de permitir calibrar."
        ),
    )

    parser.add_argument(
        "--recommended-samples",
        type=int,
        default=25,
        help=(
            "Cantidad recomendada mostrada "
            "en pantalla."
        ),
    )

    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help=(
            "Escala SOLO para visualización. "
            "La calibración utiliza resolución original."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./calibration_output",
        help="Carpeta de salida.",
    )

    return parser.parse_args()


# ============================================================
# CÁMARA
# ============================================================


def parse_camera_source(
    camera: str,
) -> Union[int, str]:
    """
    '0' -> 0
    '1' -> 1
    '/dev/video0' -> '/dev/video0'
    """

    if camera.isdigit():
        return int(camera)

    return camera


def fourcc_to_string(
    fourcc_int: int,
) -> str:
    return "".join(
        chr(
            (fourcc_int >> (8 * i))
            & 0xFF
        )
        for i in range(4)
    )


def open_camera(args):
    source = parse_camera_source(
        args.camera
    )

    print()
    print(
        "Abriendo cámara:",
        source,
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara {source}"
        )

    # FOURCC antes de resolución/FPS.
    requested_fourcc = (
        args.fourcc.upper()
    )

    if len(requested_fourcc) != 4:
        raise ValueError(
            "--fourcc debe tener exactamente "
            "4 caracteres. Ejemplo: MJPG"
        )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *requested_fourcc
        ),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        args.fps,
    )

    # No todos los backends respetan esto,
    # pero intentamos reducir buffering.
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    actual_fourcc_int = int(
        cap.get(
            cv2.CAP_PROP_FOURCC
        )
    )

    actual_fourcc = fourcc_to_string(
        actual_fourcc_int
    )

    print()
    print(
        "========================================"
    )
    print(
        "             CÁMARA ABIERTA"
    )
    print(
        "========================================"
    )
    print(
        f"Source:     {source}"
    )
    print(
        f"Solicitado: "
        f"{args.width}x{args.height} "
        f"@ {args.fps:.2f} FPS"
    )
    print(
        f"Obtenido:   "
        f"{actual_width}x{actual_height} "
        f"@ {actual_fps:.2f} FPS"
    )
    print(
        f"FOURCC:     {actual_fourcc}"
    )
    print(
        f"Vista:      {args.view}"
    )
    print(
        "========================================"
    )
    print()

    if (
        actual_width != args.width
        or actual_height != args.height
    ):
        print(
            "[WARNING] La cámara NO aceptó "
            "exactamente la resolución solicitada."
        )
        print(
            "La calibración corresponderá a la "
            "resolución que realmente entregue."
        )
        print()

    return cap


# ============================================================
# SELECCIÓN MONO / ESTÉREO
# ============================================================


def select_camera_view(
    frame: np.ndarray,
    view: str,
) -> np.ndarray:
    """
    mono:
        devuelve frame completo.

    left:
        asume LEFT | RIGHT y devuelve
        mitad izquierda.

    right:
        asume LEFT | RIGHT y devuelve
        mitad derecha.
    """

    if view == "mono":
        return frame

    height, width = frame.shape[:2]

    if width % 2 != 0:
        raise ValueError(
            "Para usar --view left/right "
            "el ancho del frame debe ser par. "
            f"Recibido: {width}"
        )

    half_width = width // 2

    if view == "left":
        return frame[
            :,
            :half_width,
        ]

    if view == "right":
        return frame[
            :,
            half_width:,
        ]

    raise ValueError(
        f"Vista desconocida: {view}"
    )


# ============================================================
# CHECKERBOARD
# ============================================================


def create_object_points(
    board_cols: int,
    board_rows: int,
    square_size_m: float,
) -> np.ndarray:
    """
    Coordenadas físicas conocidas del tablero.

    Ejemplo con cuadrados de 20 mm:

    (0.00, 0.00, 0)
    (0.02, 0.00, 0)
    (0.04, 0.00, 0)
    ...

    El tablero se considera plano:
        Z = 0
    """

    points = np.zeros(
        (
            board_rows * board_cols,
            3,
        ),
        dtype=np.float32,
    )

    points[:, :2] = (
        np.mgrid[
            0:board_cols,
            0:board_rows,
        ]
        .T
        .reshape(-1, 2)
    )

    points *= square_size_m

    return points


def detect_checkerboard(
    frame_bgr: np.ndarray,
    board_cols: int,
    board_rows: int,
):
    gray = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    pattern_size = (
        board_cols,
        board_rows,
    )

    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )

    found, corners = (
        cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=flags,
        )
    )

    image_size = (
        gray.shape[1],
        gray.shape[0],
    )

    return (
        found,
        corners,
        image_size,
    )


# ============================================================
# ERROR DE REPROYECCIÓN
# ============================================================


def calculate_reprojection_errors(
    object_points,
    image_points,
    rvecs,
    tvecs,
    camera_matrix,
    distortion,
):
    """
    Calcula el error medio Euclídeo en píxeles
    para cada imagen y el promedio global.
    """

    per_image_errors = []

    for i in range(
        len(object_points)
    ):
        projected, _ = (
            cv2.projectPoints(
                object_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                distortion,
            )
        )

        observed = image_points[
            i
        ].reshape(-1, 2)

        predicted = projected.reshape(
            -1,
            2,
        )

        distances = np.linalg.norm(
            observed - predicted,
            axis=1,
        )

        mean_error = float(
            np.mean(distances)
        )

        per_image_errors.append(
            mean_error
        )

    global_mean = float(
        np.mean(per_image_errors)
    )

    return (
        global_mean,
        per_image_errors,
    )


# ============================================================
# GUARDAR CALIBRACIÓN
# ============================================================


def save_calibration(
    output_file: Path,
    args,
    camera_matrix,
    distortion,
    image_size,
    rms,
    mean_error,
    per_image_errors,
    num_samples,
):
    fx = float(
        camera_matrix[0, 0]
    )

    fy = float(
        camera_matrix[1, 1]
    )

    cx = float(
        camera_matrix[0, 2]
    )

    cy = float(
        camera_matrix[1, 2]
    )

    data = {
        "camera": {
            "source": args.camera,
            "view": args.view,
            "fourcc": args.fourcc,
            "capture_width_requested": (
                int(args.width)
            ),
            "capture_height_requested": (
                int(args.height)
            ),
            "capture_fps_requested": (
                float(args.fps)
            ),
        },
        "calibrated_image": {
            "width": int(
                image_size[0]
            ),
            "height": int(
                image_size[1]
            ),
        },
        "checkerboard": {
            "inner_corners_cols": (
                int(args.board_cols)
            ),
            "inner_corners_rows": (
                int(args.board_rows)
            ),
            "square_size_mm": (
                float(args.square_mm)
            ),
            "square_size_m": (
                float(
                    args.square_mm
                    / 1000.0
                )
            ),
        },
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "camera_matrix": (
            camera_matrix.tolist()
        ),
        "distortion_coefficients": (
            distortion
            .reshape(-1)
            .tolist()
        ),
        "calibration_quality": {
            "num_samples": int(
                num_samples
            ),
            "opencv_rms_px": float(
                rms
            ),
            "mean_reprojection_error_px": (
                float(mean_error)
            ),
            "per_image_error_px": [
                float(value)
                for value
                in per_image_errors
            ],
        },
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )

    return (
        fx,
        fy,
        cx,
        cy,
    )


# ============================================================
# CALIBRAR
# ============================================================


def calibrate(
    args,
    object_points,
    image_points,
    image_size,
    output_file,
):
    print()
    print(
        "Calculando calibración..."
    )

    (
        rms,
        camera_matrix,
        distortion,
        rvecs,
        tvecs,
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    (
        mean_error,
        per_image_errors,
    ) = calculate_reprojection_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        distortion,
    )

    (
        fx,
        fy,
        cx,
        cy,
    ) = save_calibration(
        output_file,
        args,
        camera_matrix,
        distortion,
        image_size,
        rms,
        mean_error,
        per_image_errors,
        len(image_points),
    )

    print()
    print(
        "========================================"
    )
    print(
        "          RESULTADO CALIBRACIÓN"
    )
    print(
        "========================================"
    )

    print(
        f"Vista calibrada: "
        f"{args.view.upper()}"
    )

    print(
        f"Resolución calibrada: "
        f"{image_size[0]}x{image_size[1]}"
    )

    print(
        f"Capturas utilizadas: "
        f"{len(image_points)}"
    )

    print()

    print(
        f"fx = {fx:.6f}"
    )

    print(
        f"fy = {fy:.6f}"
    )

    print(
        f"cx = {cx:.6f}"
    )

    print(
        f"cy = {cy:.6f}"
    )

    print()

    print(
        "Matriz intrínseca K:"
    )

    print(
        camera_matrix
    )

    print()

    print(
        "Coeficientes de distorsión:"
    )

    print(
        distortion.reshape(-1)
    )

    print()

    print(
        f"RMS OpenCV: "
        f"{rms:.6f} px"
    )

    print(
        "Error medio de reproyección: "
        f"{mean_error:.6f} px"
    )

    print()

    print(
        "Valores para DA3Metric:"
    )

    print(
        "----------------------------------------"
    )
    print(
        "depth_mode: 'metric_focal'"
    )
    print(
        f"fx: {fx:.6f}"
    )
    print(
        f"fy: {fy:.6f}"
    )
    print(
        "----------------------------------------"
    )

    print()

    print(
        f"Guardado en:\n{output_file}"
    )

    print(
        "========================================"
    )


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    if args.board_cols <= 1:
        raise ValueError(
            "--board-cols debe ser > 1"
        )

    if args.board_rows <= 1:
        raise ValueError(
            "--board-rows debe ser > 1"
        )

    if args.square_mm <= 0:
        raise ValueError(
            "--square-mm debe ser > 0"
        )

    if args.preview_scale <= 0:
        raise ValueError(
            "--preview-scale debe ser > 0"
        )

    square_size_m = (
        args.square_mm
        / 1000.0
    )

    output_dir = Path(
        args.output
    )

    capture_dir = (
        output_dir / "images"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "camera_calibration.yaml"
    )

    cap = open_camera(args)

    object_template = (
        create_object_points(
            board_cols=args.board_cols,
            board_rows=args.board_rows,
            square_size_m=square_size_m,
        )
    )

    object_points = []
    image_points = []

    image_size = None

    print(
        "========================================"
    )
    print(
        "           CHECKERBOARD"
    )
    print(
        "========================================"
    )

    print(
        "Esquinas internas: "
        f"{args.board_cols} x "
        f"{args.board_rows}"
    )

    print(
        "Cuadrados impresos esperados: "
        f"{args.board_cols + 1} x "
        f"{args.board_rows + 1}"
    )

    print(
        f"Tamaño cuadrado: "
        f"{args.square_mm:.3f} mm"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Controles:"
    )
    print(
        "  SPACE : guardar captura"
    )
    print(
        "  D     : eliminar última captura"
    )
    print(
        "  C     : calcular calibración"
    )
    print(
        "  Q/ESC : salir"
    )

    print()

    print(
        f"Mínimo: {args.min_samples}"
    )

    print(
        "Recomendado: "
        f"{args.recommended_samples}"
    )

    print()

    first_frame = True

    try:
        while True:
            success, full_frame = (
                cap.read()
            )

            if not success:
                print(
                    "[ERROR] No se pudo "
                    "capturar un frame."
                )
                break

            camera_frame = (
                select_camera_view(
                    full_frame,
                    args.view,
                )
            )

            if first_frame:
                full_h, full_w = (
                    full_frame.shape[:2]
                )

                view_h, view_w = (
                    camera_frame.shape[:2]
                )

                print(
                    "Primer frame:"
                )

                print(
                    "  Frame recibido: "
                    f"{full_w}x{full_h}"
                )

                print(
                    "  Imagen usada para "
                    "calibración: "
                    f"{view_w}x{view_h}"
                )

                print()

                first_frame = False

            display = (
                camera_frame.copy()
            )

            (
                found,
                corners,
                current_image_size,
            ) = detect_checkerboard(
                camera_frame,
                args.board_cols,
                args.board_rows,
            )

            image_size = (
                current_image_size
            )

            if found:
                cv2.drawChessboardCorners(
                    display,
                    (
                        args.board_cols,
                        args.board_rows,
                    ),
                    corners,
                    found,
                )

                detection_text = (
                    "CHECKERBOARD DETECTADO"
                )

                detection_color = (
                    0,
                    255,
                    0,
                )

            else:
                detection_text = (
                    "CHECKERBOARD NO DETECTADO"
                )

                detection_color = (
                    0,
                    0,
                    255,
                )

            cv2.putText(
                display,
                detection_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                detection_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Capturas: "
                    f"{len(image_points)}"
                    f"/{args.recommended_samples}"
                ),
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (
                    255,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Vista: {args.view.upper()} | "
                    f"{image_size[0]}x"
                    f"{image_size[1]}"
                ),
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    "SPACE=guardar | "
                    "D=borrar | "
                    "C=calibrar | "
                    "Q=salir"
                ),
                (30, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            preview = cv2.resize(
                display,
                None,
                fx=args.preview_scale,
                fy=args.preview_scale,
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(
                "Camera Calibration",
                preview,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ----------------------------------------------
            # SPACE -> guardar
            # ----------------------------------------------

            if key == 32:
                if not found:
                    print(
                        "[WARN] No se guardó: "
                        "el tablero no fue detectado."
                    )
                    continue

                object_points.append(
                    object_template.copy()
                )

                image_points.append(
                    corners.copy()
                )

                index = len(
                    image_points
                )

                image_path = (
                    capture_dir
                    / f"calib_{index:03d}.png"
                )

                cv2.imwrite(
                    str(image_path),
                    camera_frame,
                )

                print(
                    f"[OK] Captura {index} guardada."
                )

            # ----------------------------------------------
            # D -> borrar última
            # ----------------------------------------------

            elif key in (
                ord("d"),
                ord("D"),
            ):
                if not image_points:
                    print(
                        "[WARN] No hay capturas "
                        "para eliminar."
                    )
                    continue

                image_points.pop()
                object_points.pop()

                print(
                    "[OK] Última captura eliminada."
                )

            # ----------------------------------------------
            # C -> calibrar
            # ----------------------------------------------

            elif key in (
                ord("c"),
                ord("C"),
            ):
                if (
                    len(image_points)
                    < args.min_samples
                ):
                    print(
                        "[WARN] Solo tienes "
                        f"{len(image_points)} capturas."
                    )

                    print(
                        "Necesitas al menos "
                        f"{args.min_samples}."
                    )

                    continue

                calibrate(
                    args=args,
                    object_points=object_points,
                    image_points=image_points,
                    image_size=image_size,
                    output_file=output_file,
                )

                break

            # ----------------------------------------------
            # Q / ESC
            # ----------------------------------------------

            elif key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()