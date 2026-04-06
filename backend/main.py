"""
Sketch-to-Image GAN — FastAPI Backend
Serves the trained GAN model for inference with GPU acceleration.
"""

import sys
import os
import io
import base64
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# ── Add parent directory to path so we can import model classes ──
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(BACKEND_DIR, "..")
sys.path.insert(0, PARENT_DIR)

from SketchEncoder import SketchEncoder
from ClipEncoder import ClipEncoder
from Generator import Generator
from Discriminator import LiteMultiModalDiscriminator
from SketchGAN import SketchGAN

# ── Global model references ──
sketch_encoder = None
clip_encoder = None
generator = None
device = None


def setup_gpu():
    """Configure GPU for optimal inference performance."""
    global device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        print(f"GPU Detected: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"CUDA Version: {torch.version.cuda}")

        # Enable TF32 for Ampere+ GPUs (RTX 30xx / 40xx) — major speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Set float32 matmul precision
        torch.set_float32_matmul_precision("high")

        print("TF32 enabled | cuDNN benchmark ON")
    else:
        device = torch.device("cpu")
        print("No GPU found — using CPU (inference will be slow)")

    return device


def find_checkpoint():
    """Find the best checkpoint file in the backend directory."""
    # Look for .ckpt files in the backend root
    ckpt_files = glob.glob(os.path.join(BACKEND_DIR, "*.ckpt"))

    if not ckpt_files:
        # Fallback: check parent checkpoints folder
        ckpt_files = glob.glob(os.path.join(PARENT_DIR, "checkpoints", "*.ckpt"))

    if not ckpt_files:
        return None

    # Prefer 'last.ckpt' if it exists, otherwise pick the first one found
    for f in ckpt_files:
        if os.path.basename(f) == "last.ckpt":
            return f

    # Return the most recently modified checkpoint
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    return ckpt_files[0]


def load_models():
    """Load all model components for inference with GPU acceleration."""
    global sketch_encoder, clip_encoder, generator, device

    device = setup_gpu()

    # ── Initialize model architectures ──
    print("Initializing models...")
    sketch_enc = SketchEncoder(latent_dim=512)
    gen = Generator(sketch_latent_dim=512, text_dim=768)
    dis = LiteMultiModalDiscriminator(text_dim=768)

    clip_encoder = ClipEncoder(model_name="distilbert-base-uncased")  # self-places on device

    # ── Find and load checkpoint ──
    ckpt_path = find_checkpoint()

    if ckpt_path:
        print(f"\nLoading checkpoint: {os.path.basename(ckpt_path)}")
        print(f"Path: {ckpt_path}")
        try:
            # Use PyTorch Lightning's native load_from_checkpoint
            # This ensures EMA weights, BatchNorm stats, and all state dicts are 100% properly mapped.
            lightning_model = SketchGAN.load_from_checkpoint(
                ckpt_path,
                sketch_encoder=sketch_enc,
                generator=gen,
                discriminator=dis,
                map_location=device
            )
            
            # Extract the fully loaded sub-models
            sketch_encoder = lightning_model.sketch_encoder.to(device)
            # Prioritize EMA generator for inference
            generator = lightning_model.generator_ema.to(device)
            
            print("Models successfully loaded from Lightning checkpoint")

        except Exception as e:
            print(f"Checkpoint load failed: {e}")
            print("Using randomly initialized weights (demo mode)")
            sketch_encoder = sketch_enc.to(device)
            generator = gen.to(device)
    else:
        print("\nNo checkpoint found in backend/ or checkpoints/")
        print("Running with random weights (demo mode)")
        sketch_encoder = sketch_enc.to(device)
        generator = gen.to(device)

    # ── Set to eval mode ──
    sketch_encoder.eval()
    generator.eval()

    # ── Use half precision on GPU for faster inference ──
    if device.type == "cuda":
        sketch_encoder = sketch_encoder.half()
        generator = generator.half()
        print("\n Models converted to FP16 for faster GPU inference")

    # ── GPU memory summary ──
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU Memory: {allocated:.0f} MB allocated / {reserved:.0f} MB reserved")

    print("\nAll models loaded and ready!\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    load_models()
    yield
    # Cleanup GPU memory on shutdown
    if device and device.type == "cuda":
        torch.cuda.empty_cache()
    print("Shutting down...")


# ══════════════════════════════════════════════════════════════
#  FastAPI App
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="SketchGAN API",
    description="Generate realistic images from sketches and text descriptions using a GAN",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow frontend to connect ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Preprocessing transforms (must match training) ──
sketch_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(256),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),  # → [-1, 1]
])


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a [-1,1] image tensor to a base64-encoded PNG string."""
    # Always convert back to float32 for PIL
    img = tensor.squeeze(0).cpu().float().clamp(-1, 1)
    img = (img + 1) / 2  # → [0, 1]
    img = T.ToPILImage()(img)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check endpoint — reports device, GPU info, and model status."""
    gpu_info = None
    if device and device.type == "cuda":
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 1),
            "memory_reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 1),
        }

    return {
        "status": "ok",
        "device": str(device),
        "gpu": gpu_info,
        "models_loaded": all(m is not None for m in [sketch_encoder, clip_encoder, generator]),
    }


@app.post("/generate")
async def generate(
    sketch: UploadFile = File(None),
    sketch_base64: str = Form(None),
    description: str = Form(...),
):
    """
    Generate an image from a sketch + text description.

    Accepts the sketch as either:
      - A file upload (sketch field)
      - A base64-encoded PNG string (sketch_base64 field, from the drawing canvas)
    """
    global sketch_encoder, clip_encoder, generator, device

    # ── 1. Parse the sketch input ──
    try:
        if sketch is not None and sketch.filename:
            raw_bytes = await sketch.read()
            sketch_image = Image.open(io.BytesIO(raw_bytes)).convert("L")
        elif sketch_base64:
            if "," in sketch_base64:
                sketch_base64 = sketch_base64.split(",", 1)[1]
            raw_bytes = base64.b64decode(sketch_base64)
            sketch_image = Image.open(io.BytesIO(raw_bytes)).convert("L")
        else:
            raise HTTPException(
                status_code=400,
                detail="No sketch provided. Upload a file or send base64 data.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read sketch: {str(e)}")

    # ── 2. Validate description ──
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="Text description cannot be empty.")

    # ── 3. Run inference with GPU acceleration ──
    try:
        with torch.no_grad():
            # Preprocess sketch
            sketch_tensor = sketch_transform(sketch_image).unsqueeze(0).to(device)

            # Cast to FP16 if on GPU (models are already half)
            if device.type == "cuda":
                sketch_tensor = sketch_tensor.half()

            # Encode text (ClipEncoder handles its own device placement)
            z_text = clip_encoder([description.strip()])  # (1, 768)

            # Cast text embedding to FP16 if on GPU
            if device.type == "cuda":
                z_text = z_text.half()

            # Run through the pipeline with autocast for mixed precision
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    z_sketch = sketch_encoder(sketch_tensor)  # (1, 512, 32, 32)
                    generated = generator(z_sketch, z_text)   # (1, 3, 256, 256)
            else:
                z_sketch = sketch_encoder(sketch_tensor)
                generated = generator(z_sketch, z_text)

        # Convert to base64
        image_b64 = tensor_to_base64(generated)

        return JSONResponse(content={
            "success": True,
            "image": f"data:image/png;base64,{image_b64}",
            "description": description.strip(),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


#  Run directly

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
