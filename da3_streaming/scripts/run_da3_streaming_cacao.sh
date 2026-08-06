#!/usr/bin/env bash

set -Eeuo pipefail

# ------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------

STREAM_DIR="${DA3_STREAM_DIR:-$HOME/Documents/GitHub/Depth-Anything-3/da3_streaming}"
STREAM_SCRIPT="${DA3_STREAM_SCRIPT:-$STREAM_DIR/da3_streaming_cacao.py}"

IMAGE_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
MAX_FRAMES="${3:-60}"
CONFIG_FILE="${4:-$STREAM_DIR/configs/cacao_da3_streaming_hq.yaml}"

show_usage() {
    cat <<EOF
Uso:

  $0 IMAGE_DIR OUTPUT_DIR [MAX_FRAMES] [CONFIG_FILE]

Argumentos:

  IMAGE_DIR      Carpeta con frames JPG, JPEG o PNG.
  OUTPUT_DIR     Carpeta donde guardar resultados.
  MAX_FRAMES     Número máximo de frames.
                 Use 0 para procesar toda la secuencia.
                 Valor predeterminado: 60.
  CONFIG_FILE    YAML de configuración.
                 Predeterminado:
                 $STREAM_DIR/configs/cacao_da3_streaming_hq.yaml

Ejemplo de prueba:

  $0 \
    "\$HOME/Documents/GitHub/ag-tracking/data/CacaoMOT/MOT/seq008/img" \
    "\$HOME/Documents/GitHub/ag-tracking/output/CacaoMOT/DA3/seq008_da3stream_p1008_inputres_n60" \
    60

Ejemplo de secuencia completa:

  $0 \
    "\$HOME/Documents/GitHub/ag-tracking/data/CacaoMOT/MOT/seq008/img" \
    "\$HOME/Documents/GitHub/ag-tracking/output/CacaoMOT/DA3/seq008_da3stream_p1008_inputres_full" \
    0
EOF
}

if [[ -z "$IMAGE_DIR" || -z "$OUTPUT_DIR" ]]; then
    show_usage
    exit 1
fi

if [[ ! "$MAX_FRAMES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MAX_FRAMES debe ser un entero mayor o igual a cero."
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: no hay un entorno venv activo."
    echo "Activa tu entorno antes de ejecutar el script."
    exit 1
fi

if [[ ! -d "$STREAM_DIR" ]]; then
    echo "ERROR: no existe DA3-Streaming:"
    echo "$STREAM_DIR"
    exit 1
fi

if [[ ! -f "$STREAM_SCRIPT" ]]; then
    echo "ERROR: no existe el script personalizado:"
    echo "$STREAM_SCRIPT"
    exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "ERROR: no existe la carpeta de imágenes:"
    echo "$IMAGE_DIR"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: no existe el YAML:"
    echo "$CONFIG_FILE"
    exit 1
fi

if [[ ! -f "$STREAM_DIR/weights/model.safetensors" ]]; then
    echo "ERROR: no se encontró:"
    echo "$STREAM_DIR/weights/model.safetensors"
    echo "Ejecuta primero: bash scripts/download_weights.sh"
    exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: la carpeta de salida ya existe:"
    echo "$OUTPUT_DIR"
    echo "Elimínala o selecciona otro nombre para evitar mezclar resultados."
    exit 1
fi

python - <<'PY'
import torch
from depth_anything_3.api import DepthAnything3

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch no detecta CUDA.")

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("DA3 importado correctamente.")
PY

IMAGE_DIR="$(realpath "$IMAGE_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
CONFIG_FILE="$(realpath "$CONFIG_FILE")"
STREAM_SCRIPT="$(realpath "$STREAM_SCRIPT")"

TEMP_DIR=""

cleanup() {
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT

if [[ "$MAX_FRAMES" -eq 0 ]]; then
    INPUT_DIR="$IMAGE_DIR"
else
    TEMP_DIR="$(mktemp -d -p /tmp da3_cacao_XXXXXX)"
    INPUT_DIR="$TEMP_DIR"

    python - "$IMAGE_DIR" "$INPUT_DIR" "$MAX_FRAMES" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
maximum = int(sys.argv[3])
valid_extensions = {".jpg", ".jpeg", ".png"}

def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]

images = sorted(
    (
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    ),
    key=natural_key,
)

if not images:
    raise RuntimeError(f"No se encontraron imágenes JPG, JPEG o PNG en {source}")

selected = images[:maximum]

for image_path in selected:
    link_path = destination / image_path.name
    link_path.symlink_to(image_path.resolve())

print("Frames encontrados:", len(images))
print("Frames seleccionados:", len(selected))
print("Directorio temporal:", destination)
PY
fi

EXPECTED_FRAMES="$(
    python - "$INPUT_DIR" <<'PY'
from pathlib import Path
import sys

folder = Path(sys.argv[1])
valid_extensions = {".jpg", ".jpeg", ".png"}
print(sum(
    1
    for path in folder.iterdir()
    if path.is_file() and path.suffix.lower() in valid_extensions
))
PY
)"

if [[ "$EXPECTED_FRAMES" -eq 0 ]]; then
    echo "ERROR: no hay frames para procesar."
    exit 1
fi

echo
echo "========================================"
echo " DA3-Streaming CacaoMOT"
echo "========================================"
echo "Entrada:       $INPUT_DIR"
echo "Salida:        $OUTPUT_DIR"
echo "Script:        $STREAM_SCRIPT"
echo "Configuración: $CONFIG_FILE"
echo "Frames:        $EXPECTED_FRAMES"
echo "Entorno:       $VIRTUAL_ENV"
echo "========================================"
echo

echo "Parámetros principales del YAML:"

python - "$CONFIG_FILE" "$INPUT_DIR" <<'PY'
from pathlib import Path
import sys
import yaml
from PIL import Image

config_path = Path(sys.argv[1])
input_dir = Path(sys.argv[2])

with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

model = config["Model"]
keys = [
    "process_res",
    "process_res_method",
    "output_at_input_resolution",
    "save_processed_results",
    "chunk_size",
    "overlap",
    "loop_enable",
    "delete_temp_files",
    "align_lib",
    "align_method",
    "save_depth_conf_result",
    "save_debug_info",
]

for key in keys:
    print(f"  {key}: {model[key]}")

if model["process_res"] <= 0:
    raise ValueError("process_res debe ser mayor que cero.")
if model["process_res_method"] not in {
    "upper_bound_resize",
    "lower_bound_resize",
}:
    raise ValueError(
        "Para salida registrada se recomienda un método '*_resize', no '*_crop'."
    )
if model["overlap"] >= model["chunk_size"]:
    raise ValueError("overlap debe ser menor que chunk_size.")

valid_extensions = {".jpg", ".jpeg", ".png"}
images = [
    path
    for path in input_dir.iterdir()
    if path.is_file() and path.suffix.lower() in valid_extensions
]

sizes = []
for path in images:
    with Image.open(path) as image:
        sizes.append((image.width, image.height))

unique_sizes = sorted(set(sizes))
if len(unique_sizes) != 1:
    raise ValueError(
        f"Todos los frames deben tener la misma resolución. Encontradas: {unique_sizes}"
    )

width, height = unique_sizes[0]
print(f"  input_resolution: {width}x{height}")
PY

mkdir -p "$(dirname "$OUTPUT_DIR")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$STREAM_DIR"

python "$STREAM_SCRIPT" \
    --image_dir "$INPUT_DIR" \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR"

NPZ_COUNT=0
if [[ -d "$OUTPUT_DIR/results_output" ]]; then
    NPZ_COUNT="$(
        find "$OUTPUT_DIR/results_output" \
            -maxdepth 1 \
            -type f \
            -name 'frame_*.npz' |
        wc -l
    )"
fi

POSE_COUNT=0
if [[ -f "$OUTPUT_DIR/camera_poses.txt" ]]; then
    POSE_COUNT="$(wc -l < "$OUTPUT_DIR/camera_poses.txt")"
fi

INTRINSICS_FILE="$OUTPUT_DIR/camera_intrinsics.txt"
if [[ ! -f "$INTRINSICS_FILE" ]]; then
    INTRINSICS_FILE="$OUTPUT_DIR/intrinsic.txt"
fi

INTRINSIC_COUNT=0
if [[ -f "$INTRINSICS_FILE" ]]; then
    INTRINSIC_COUNT="$(wc -l < "$INTRINSICS_FILE")"
fi

echo
echo "========================================"
echo " Validación"
echo "========================================"
echo "Frames esperados: $EXPECTED_FRAMES"
echo "Archivos NPZ:     $NPZ_COUNT"
echo "Poses:            $POSE_COUNT"
echo "Intrínsecos:      $INTRINSIC_COUNT"
echo "Salida:           $OUTPUT_DIR"
echo "========================================"

if [[ "$NPZ_COUNT" -ne "$EXPECTED_FRAMES" ]]; then
    echo "ERROR: el número de NPZ no coincide con los frames."
    exit 1
fi
if [[ "$POSE_COUNT" -ne "$EXPECTED_FRAMES" ]]; then
    echo "ERROR: el número de poses no coincide con los frames."
    exit 1
fi
if [[ "$INTRINSIC_COUNT" -ne "$EXPECTED_FRAMES" ]]; then
    echo "ERROR: el número de intrínsecos no coincide con los frames."
    exit 1
fi

python - "$OUTPUT_DIR/results_output" "$INPUT_DIR" <<'PY'
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image

folder = Path(sys.argv[1])
input_dir = Path(sys.argv[2])
valid_extensions = {".jpg", ".jpeg", ".png"}

def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]

files = sorted(folder.glob("frame_*.npz"), key=natural_key)
images = sorted(
    (
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    ),
    key=natural_key,
)

if not files:
    raise RuntimeError("No se encontraron archivos NPZ.")

for sample_path, input_path in ((files[0], images[0]), (files[-1], images[-1])):
    with np.load(sample_path) as sample:
        print()
        print("Resultado:", sample_path.name)
        print("Claves:", sample.files)

        required = {"image", "depth", "conf", "intrinsics"}
        missing = required.difference(sample.files)
        if missing:
            raise RuntimeError(f"Faltan claves obligatorias: {sorted(missing)}")

        with Image.open(input_path) as image:
            expected_hw = (image.height, image.width)

        depth = sample["depth"]
        confidence = sample["conf"]
        rgb = sample["image"]

        if depth.shape != expected_hw:
            raise RuntimeError(
                f"Depth {depth.shape} no coincide con entrada {expected_hw}."
            )
        if confidence.shape != expected_hw:
            raise RuntimeError(
                f"Confidence {confidence.shape} no coincide con entrada {expected_hw}."
            )
        if rgb.shape[:2] != expected_hw:
            raise RuntimeError(
                f"RGB {rgb.shape[:2]} no coincide con entrada {expected_hw}."
            )
        if not np.isfinite(depth).all():
            raise RuntimeError("Depth contiene NaN o infinitos.")
        if not np.isfinite(confidence).all():
            raise RuntimeError("Confidence contiene NaN o infinitos.")
        if float(confidence.min()) < 0.0:
            raise RuntimeError("Confidence contiene valores negativos.")

        print("Input HW:", expected_hw)
        print("Depth HW:", depth.shape)
        print("Processed HW:", tuple(sample["processed_hw"]))
        print(
            "Depth:",
            "min =", float(depth.min()),
            "max =", float(depth.max()),
            "mean =", float(depth.mean()),
        )
        print(
            "Confidence:",
            "min =", float(confidence.min()),
            "max =", float(confidence.max()),
            "mean =", float(confidence.mean()),
        )

print()
print("Validación final correcta.")
PY