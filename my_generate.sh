#MODEL_WEIGHTS_PATH="training-runs/00014-stylegan3-r-ncsg3-gpus1-batch32-gamma2/network-snapshot-000001.pkl"
MODEL_WEIGHTS_PATH="training-runs/00029-stylegan3-t-ncsg3-gpus1-batch12-gamma2/network-snapshot-001200.pkl"
OUT_DIR_PATH="generations"

mkdir -p "$OUT_DIR_PATH"

SEEDS="0"

for i in {1..100}; do
    SEEDS="$i,${SEEDS}"
done

echo "$SEEDS"

uv run gen_images.py \
    --network "$MODEL_WEIGHTS_PATH" \
    --seeds $SEEDS \
    --class 0 \
    --outdir "$OUT_DIR_PATH"

SEEDS="100"
for i in {101..200}; do
    SEEDS="$i,${SEEDS}"
done

echo "$SEEDS"

uv run gen_images.py \
    --network "$MODEL_WEIGHTS_PATH" \
    --seeds $SEEDS \
    --class 1 \
    --outdir "$OUT_DIR_PATH"
