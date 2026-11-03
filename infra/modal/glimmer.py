"""Glimmer critic on Modal: upload the SFT data, QLoRA-train Muse Glimmer 30B on 1x H100, merge, and serve base and
tuned models with vLLM's Glimmer image (master doc §3.5).

  uv run modal run infra/modal/glimmer.py::upload            # data/glimmer_sft -> volume (once)
  uv run modal run --detach infra/modal/glimmer.py::train    # QLoRA + merge, a few hours on 1x H100
  uv run modal deploy infra/modal/glimmer.py                 # two OpenAI-compatible endpoints: base and tuned
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

app = modal.App("twinproof-glimmer")
vol = modal.Volume.from_name("glimmer-weights", create_if_missing=True)

BASE = "meta-models/Muse-Glimmer-30B"  # Apache 2.0, not gated
OUT = "/vol/out/glimmer-qlora"
ENV = {"HF_HOME": "/vol/hf"}  # one weight download, shared by training and the base endpoint

# Axolotl's examples/muse-glimmer/qlora-vision.yaml with the master doc's overrides (wandb off, our data, 2 epochs)
CONFIG = f"""
base_model: {BASE}
processor_type: AutoProcessor
plugins: [axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin]
load_in_4bit: true
skip_prepare_dataset: true
remove_unused_columns: false
sample_packing: false
eval_sample_packing: false
eot_tokens: ["<|eot|>"]
datasets:
  - path: data/glimmer_sft/train_teacher.jsonl  # rows with Spark 1.3 rationales (scripts.add_glimmer_rationales)
    ds_type: json
    type: chat_template
    field_messages: messages
dataset_prepared_path: /vol/prepared
val_set_size: 0.0
output_dir: {OUT}
adapter: qlora
sequence_len: 4096
pad_to_sequence_len: false
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: '(model\\.language_model\\.layers\\.\\d+\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)|model\\.vision_tower\\.layers\\.\\d+\\.(attn\\.(q_proj|k_proj|v_proj|proj)|mlp\\.fc[12])|model\\.vision_adapter\\.fc[12]|model\\.vision_projection)'
gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 2
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 1.0e-4
bf16: auto
tf32: true
gradient_checkpointing: true
logging_steps: 5
attn_implementation: flash_attention_2
warmup_ratio: 0.1
evals_per_epoch: 0
saves_per_epoch: 1
weight_decay: 0.0
"""

axolotl = modal.Image.from_registry("axolotlai/axolotl:main-latest").entrypoint([]).env(ENV)
vllm = modal.Image.from_registry("vllm/vllm-openai:muse-glimmer").entrypoint([]).env(ENV)


@app.local_entrypoint()
def upload() -> None:
    data = Path("data/glimmer_sft")
    if not (data / "train_teacher.jsonl").exists():
        raise SystemExit("run scripts.gen_glimmer_sft, then scripts.add_glimmer_rationales")
    with vol.batch_upload(force=True) as batch:  # rows reference images as data/glimmer_sft/img/*.png
        batch.put_directory(str(data), "/data/glimmer_sft")
    print("uploaded", data)


@app.function(image=axolotl, volumes={"/vol": vol}, timeout=2 * 3600, cpu=4)
def fetch() -> None:
    """Weights onto the volume from a CPU container, so the H100 does not idle on a 60 GB download."""
    from huggingface_hub import snapshot_download
    subprocess.run(["python", "-c", "import axolotl; print('axolotl', axolotl.__version__)"], check=True)
    snapshot_download(BASE)
    vol.commit()


@app.function(image=axolotl, gpu="H100", volumes={"/vol": vol}, timeout=6 * 3600)
def train() -> None:
    Path("/vol/qlora.yaml").write_text(CONFIG)
    for cmd in (["axolotl", "train", "/vol/qlora.yaml"],
                ["axolotl", "merge-lora", "/vol/qlora.yaml", "--lora-model-dir", OUT]):
        subprocess.run(cmd, cwd="/vol", check=True)  # cwd: the jsonl image paths are relative to /vol
        vol.commit()


def _serve(model: str) -> None:
    # Popen, never wait(): a blocking web_server function never reports ready
    subprocess.Popen(["vllm", "serve", model, "--served-model-name", "glimmer-critic",
                      "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer",
                      "--max-model-len", "8192", "--quantization", "fp8", "--host", "0.0.0.0", "--port", "8000"])


SERVE = dict(image=vllm, gpu="L40S", volumes={"/vol": vol}, scaledown_window=300, max_containers=1, timeout=3600)


@app.function(**SERVE)
@modal.web_server(port=8000, startup_timeout=1200)
def serve_base() -> None:
    _serve(BASE)


@app.function(**SERVE)
@modal.web_server(port=8000, startup_timeout=1200)
def serve_tuned() -> None:
    _serve(f"{OUT}/merged")
