# api.py

import os
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import io
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
            # Extract category name from filename (e.g., "Shoes" from "best_model_Shoes.pth")
            category = filename.replace("best_model_", "").replace(".pth", "")
            
            # Initialize the model
            # Since each model was trained on a single category, num_classes is 1
            model = CVAE(img_channels=3, img_size=64, latent_dim=64, num_classes=1).to(DEVICE)
            
            # Load the saved weights
            model_path = os.path.join(CHECKPOINTS_DIR, filename)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval() # Set model to evaluation mode

            # Store the loaded model in our dictionary
            MODELS[category] = model
            print(f"Loaded model for category: '{category}'")
    print("--- Model loading complete. ---")


# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    load_models()
    yield
    # This code runs on shutdown (optional)
    MODELS.clear()
    print("--- Models cleared. ---")


# 1. Create an instance of the FastAPI class with the lifespan manager
app = FastAPI(
    title="Fashion CVAE API",
    description="An API to generate images of fashion items using a Conditional VAE.",
    version="0.1.0",
    lifespan=lifespan # <-- Add this line
)

# 2. Define a root endpoint (the main page)
@app.get("/")
async def root():
    """
    This is the root endpoint. It returns a welcome message.
    """
    return {"message": "Welcome to the CVAE Image Generation API!", "loaded_models": list(MODELS.keys())}

@app.get("/generate/")
async def generate_image(category: str):
    if category not in MODELS:
        raise HTTPException(
            status_code = 404,
            detail = f"Category '{category}' not found. Available models: {list(MODELS.keys())}"
        )
    
    print(f'Generating image for category: {category}')
    model = MODELS[category]

    with torch.no_grad():
        # Create the latent vector (z) and the label tensor (y)
        # We generate 1 image, with the latent dimension of the model
        z = torch.randn(1, model.latent_dim).to(DEVICE)
        
        # The label is always 0 since each model knows only its own category
        label = torch.tensor([0], dtype=torch.long).to(DEVICE)
        
        # Generate the image tensor
        generated_tensor = model.decode(z, label)

    # --- This is Step 5: Handle the Image Response ---
    # Create an in-memory file buffer
    buffer = io.BytesIO()
    # Save the tensor to the buffer as a PNG image
    save_image(generated_tensor, buffer, format="PNG")
    # Move the buffer's cursor to the beginning
    buffer.seek(0)
    
    # Return the image as a streaming response
    return StreamingResponse(buffer, media_type="image/png")
