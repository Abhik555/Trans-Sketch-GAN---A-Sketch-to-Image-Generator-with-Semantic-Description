# Trans-SketchGAN — A-Sketch-to-Image-Generator-with-Semantic-Description

A multi-modal GAN that turns rough face sketches into photorealistic images, steered by natural-language descriptions. Built on **PyTorch Lightning** and trained on the **MM-CelebA-HQ** dataset.

## What This Project Does

Given a grayscale face sketch and a short text caption (e.g. *"a woman with brown hair and blue eyes"*), the model generates a 256×256 colour photograph that respects both the structural layout of the sketch and the semantic content of the text.

The pipeline is broken into three learnable pieces:

| Component | Role | Input → Output |
|---|---|---|
| **SketchEncoder** | Extracts spatial features from the sketch | `(B, 1, 256, 256)` → `(B, 512, 32, 32)` |
| **Generator** | Synthesises the colour image using AdaIN-based text conditioning | spatial features + text embedding → `(B, 3, 256, 256)` |
| **Discriminator** | Judges realism & text/sketch alignment (projection discriminator) | image + sketch + text → scalar score |

Text embeddings come from a **frozen DistilBERT** (`distilbert-base-uncased`), precomputed and cached to disk so the language model never touches the training loop.

## Architecture Details

### Sketch Encoder (`SketchEncoder.py`)

A lightweight ResNet-style encoder that progressively downsamples grayscale sketches while preserving spatial structure.

```
Input (1 × 256 × 256)
  │
  ├── 7×7 Conv → BN → LeakyReLU        (64 × 256 × 256)
  ├── ResBlock(64  → 128, stride=2)     (128 × 128 × 128)
  ├── ResBlock(128 → 256, stride=2)     (256 ×  64 ×  64)
  ├── ResBlock(256 → 512, stride=2)     (512 ×  32 ×  32)
  └── 3×3 Conv → BN → LeakyReLU        (512 ×  32 ×  32)  ← spatial latent
```

Each residual block uses BatchNorm + LeakyReLU(0.2) and includes a 1×1 shortcut when channel or spatial dimensions change.

### Generator (`Generator.py`)

Progressively upsamples the 32×32 spatial latent back to 256×256 RGB. Text conditioning is injected at every resolution through **Adaptive Instance Normalisation (AdaIN)** — the text embedding produces per-channel scale and shift parameters that modulate the feature maps.

```
z_sketch (512 × 32 × 32) + z_text (768)
  │
  ├── ResBlockAdaIN(512)                 (512 × 32  × 32)
  ├── Upsample → Conv → ResBlockAdaIN   (256 × 64  × 64)
  ├── Upsample → Conv → ResBlockAdaIN   (128 × 128 × 128)
  ├── Upsample → Conv → ResBlockAdaIN   (64  × 256 × 256)
  └── 3×3 Conv → Tanh                   (3   × 256 × 256)  ← output image
```

Upsampling is bilinear (not transposed convolution) to reduce checkerboard artefacts.

### Discriminator (`Discriminator.py`)

A **projection discriminator** with spectral normalisation on every weight matrix. It receives the concatenation of the real/fake image and the sketch (4 channels) and produces both an unconditional realism score and a conditional text-alignment score.

```
[image ∥ sketch] (4 × 256 × 256)
  │
  ├── SN-Conv → LeakyReLU               (64 × 256 × 256)
  ├── LiteDiscBlock(64  → 128, ↓2)      (128 × 128 × 128)
  ├── LiteDiscBlock(128 → 256, ↓2)      (256 ×  64 ×  64)
  ├── LiteDiscBlock(256 → 512, ↓2)      (512 ×  32 ×  32)
  ├── LiteDiscBlock(512 → 512, ↓2)      (512 ×  16 ×  16)
  ├── LiteDiscBlock(512 → 512, no ↓)    (512 ×  16 ×  16)
  └── Global AvgPool → Linear(1) + proj(text → 512)
```

The final score is `D_uncond(φ) + ⟨φ, proj(z_text)⟩`, following the projection discriminator formulation from Miyato & Koyama (2018).

### Text Encoder (`ClipEncoder.py`)

Wraps HuggingFace's DistilBERT. All weights are frozen — it's only used for inference. The `[CLS]` token from the last hidden state serves as the 768-d sentence embedding. Embeddings are precomputed once with `precompute_embeddings.py` and saved as `.pt` files.

## Dataset

**MM-CelebA-HQ** — a multimodal extension of CelebA-HQ containing:

- **30,000** face images at high resolution
- Corresponding **face sketches** 
- **Text captions** describing facial attributes

The dataset is expected at `./Dataset/MM-CELEBA-HQ/` with subdirectories:

```
MM-CELEBA-HQ/
├── images/              # colour face photos (.jpg)
├── sketch/sketch/       # grayscale face sketches (.jpg)
├── text/celeba-caption/ # text captions (.txt)
└── text_embeddings/     # precomputed DistilBERT embeddings (.pt)
```

The dataloader (`SketchDataset.py`) uses multithreaded scanning on first run to verify which samples have all three modalities present, then caches the valid ID list to `dataset_cache.json` for fast subsequent loads. Images are resized to 256×256 and normalised to [-1, 1].


## Training Pipeline

Training is managed by PyTorch Lightning in `train.py` with manual optimisation (required for GAN alternating updates).

### Key Training Details

| Hyperparameter | Value |
|---|---|
| Image resolution | 256 × 256 |
| Batch size | 8 |
| Optimiser | Adam (lr=1e-4, β₁=0.0, β₂=0.9) |
| GAN loss | Hinge loss |
| Reconstruction loss | L1 (λ=10) |
| Precision | FP16 mixed precision |
| Gradient accumulation | 2 steps |
| Gradient clipping | Max norm 1.0 |
| EMA decay | 0.999 |
| Max epochs | 80 |
| Early stopping | Patience 15 (monitoring FID) |
| Train/Val split | 90/10 |
| DataLoader workers | 8 (persistent, pinned memory) |

### What Happens During Training

1. **Discriminator update** — real images scored high, detached fakes scored low (hinge loss). Gradients are accumulated over 2 micro-batches before stepping.
2. **Generator update** — fake images passed back through D to get the adversarial signal, plus a pixel-level L1 loss against the ground-truth photo.
3. **EMA update** — after every accumulated step, an exponential moving average copy of the generator is updated (decay 0.999). This EMA generator is used for validation and image logging.
4. **TensorBoard logging** — every 200 steps, a side-by-side grid (sketch → generated → real) is logged for visual inspection.

### Validation

At the end of each epoch, the model computes **Fréchet Inception Distance (FID)** on the validation split using `torchmetrics.image.fid.FrechetInceptionDistance`. Checkpoints are saved based on best FID, and early stopping halts training if FID hasn't improved for 15 epochs.

### Current Results

The model has been trained for **72+ epochs** (~3,750 batches per epoch). Best FID scores from saved checkpoints:

| Checkpoint | FID Score |
|---|---|
| epoch 70 | **23.38** |
| epoch 69 | 23.39 |
| epoch 72 | 23.44 |
| epoch 61 | 24.36 |


## Setup

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU (trained on an RTX 40-series with TF32 enabled)

### Installation

```bash
# clone the repo
git clone https://github.com/<your-username>/SketchGAN.git
cd SketchGAN

# create a virtual environment (uv or pip)
uv sync          # if using uv (recommended)
# OR
pip install -e .
```

Dependencies (from `pyproject.toml`):

```
lightning >= 2.6.1
torch >= 2.10.0
torchaudio >= 2.10.0
torchvision >= 0.25.0
transformers >= 5.1.0
```

### Preparing the Dataset

1. Download the MM-CelebA-HQ dataset and extract it to `./Dataset/MM-CELEBA-HQ/`.
2. Precompute text embeddings (only needs to run once):

```bash
python precompute_embeddings.py
```

This tokenises every caption through DistilBERT and saves the 768-d `[CLS]` embeddings as individual `.pt` files under `Dataset/MM-CELEBA-HQ/text_embeddings/`.

### Training

```bash
python train.py
```

Training will automatically resume from `checkpoints/last.ckpt` if one exists. Monitor progress with TensorBoard:

```bash
tensorboard --logdir tb_logs
```


## Technical Notes

- **Why DistilBERT instead of CLIP?** DistilBERT is significantly lighter than CLIP's text encoder while still providing strong semantic embeddings. Since the captions are short attribute descriptions (not open-vocabulary), the extra capacity of CLIP isn't needed and the lower memory footprint lets us keep batch sizes reasonable on a single GPU.

- **Why precompute embeddings?** The text encoder is completely frozen during training — there's no reason to run it every epoch. Precomputing saves ~2-3 GB of GPU memory and removes the tokeniser from the data pipeline entirely.

- **Why hinge loss?** Hinge loss tends to be more stable than the original minimax or Wasserstein formulations for high-resolution image generation. Combined with spectral normalisation in the discriminator, it keeps training from diverging without needing gradient penalty.

- **Why EMA?** Exponential moving average of generator weights produces smoother, higher-quality outputs at inference time. The EMA copy is what gets evaluated during validation and is the recommended model for deployment.

- **TF32 precision** is explicitly enabled for RTX 40-series GPUs (`torch.backends.cuda.matmul.allow_tf32 = True`), which gives near-FP32 accuracy at significantly higher throughput.