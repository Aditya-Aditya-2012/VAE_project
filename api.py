import os
import io
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from torchvision.utils import save_image

# --- Import your CVAE model class ---
from cvae_model import CVAE

# --- Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINTS_DIR = "checkpoints"
MODELS = {} # A dictionary to hold our loaded models

# --- Model Loading Logic ---
def load_models():
    """
    Scans the checkpoints directory, loads all trained models,
    and stores them in the global MODELS dictionary.
    """
    print("--- Loading models... ---")
    for filename in os.listdir(CHECKPOINTS_DIR):
        if filename.startswith("best_model_") and filename.endswith(".pth"):
            category = filename.replace("best_model_", "").replace(".pth", "")
            model = CVAE(img_channels=3, img_size=64, latent_dim=64, num_classes=1).to(DEVICE)
            model_path = os.path.join(CHECKPOINTS_DIR, filename)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            MODELS[category] = model
            print(f"Loaded model for category: '{category}'")
    print("--- Model loading complete. ---")


# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    MODELS.clear()
    print("--- Models cleared. ---")


# --- Create FastAPI app instance ---
app = FastAPI(
    title="Fashion CVAE API",
    description="An API to generate images of fashion items using a Conditional VAE.",
    version="0.1.0",
    lifespan=lifespan
)

# --- 1. Add CORS Middleware ---
# This allows our frontend (even when served from the same origin)
# to make requests to our API endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 2. Modify the root endpoint to serve the HTML file ---
@app.get("/", response_class=FileResponse)
async def read_index():
    """
    Serves the main frontend HTML file.
    """
    return "index.html"
    
# --- Endpoint to get available models ---
@app.get("/api/models")
async def get_models():
    """
    Returns a list of the successfully loaded model categories.
    """
    return {"loaded_models": list(MODELS.keys())}


# --- Image Generation Endpoint ---
@app.get("/api/generate/")
async def generate_image(category: str):
    """
    Generates an image for a given category.
    """
    if category not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found. Available models: {list(MODELS.keys())}"
        )

    print(f"Generating image for category: {category}")
    model = MODELS[category]
    
    with torch.no_grad():
        z = torch.randn(1, model.latent_dim).to(DEVICE)
        label = torch.tensor([0], dtype=torch.long).to(DEVICE)
        generated_tensor = model.decode(z, label)

    buffer = io.BytesIO()
    save_image(generated_tensor, buffer, format="PNG")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")