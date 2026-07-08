MODEL_WEIGHTS_PATH="training-runs/00013-stylegan3-t-afhqv2-512x512-gpus1-batch32-gamma8.2/network-snapshot-000001.pkl"
OUT_DIR_PATH="generations"

mkdir -p "$OUT_DIR_PATH"

uv run gen_images.py \
    --network "$MODEL_WEIGHTS_PATH" \
    --seeds 1 \
    --outdir "$OUT_DIR_PATH"
