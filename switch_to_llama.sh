#!/bin/bash
set -e

MODELS_DIR="/home/hilleloh/app/models"
OLD_MODEL="$MODELS_DIR/qwen2.5-0.5b-instruct-q3_k_m.gguf"
NEW_MODEL="Llama-3.2-1B-Instruct-Q4_K_M.gguf"

echo "==> Deleting old Qwen model..."
if [ -f "$OLD_MODEL" ]; then
    rm "$OLD_MODEL"
    echo "    Deleted: $OLD_MODEL"
else
    echo "    Not found (already deleted?): $OLD_MODEL"
fi

echo "==> Downloading Llama-3.2-1B-Instruct Q3_K_M..."
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
    "$NEW_MODEL" \
    --local-dir "$MODELS_DIR"

echo "==> Done! Model saved to: $MODELS_DIR/$NEW_MODEL"
ls -lh "$MODELS_DIR"
