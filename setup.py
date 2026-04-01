import os
# Kích hoạt hf_transfer để đạt tốc độ tối đa (viết bằng Rust)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

# Tải toàn bộ folder/dataset
snapshot_download(
    repo_id="nguyenhieu3205xt/XTTS-v2-Checkpoint", # nguyenhiext/your-dataset-name
    repo_type="model", # hoặc "model"
    local_dir="./checkpoints",
    token=""
)