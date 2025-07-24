import torch
import argparse
import sys
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# --- Import your custom classes ---
from cvae_model import CVAE
from fashion_dataset import FashionDataset

def generate_images(category_name):
    """
    Generates and saves 4 images for a given subCategory using a pre-trained CVAE model.
    """
    # ---- Configuration ---- #
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    IMG_SIZE = 64
    LATENT_DIM = 64
    NUM_SAMPLES = 4  # Generate 4 images as requested
    CHECKPOINT_PATH = "/Users/shaileshrajan/Desktop/ml_proj/VAE_project/checkpoints/best_model.pth"
    CSV_PATH = "/Users/shaileshrajan/Desktop/ml_proj/VAE_project/styles_cleaned.csv"
    IMG_DIR = "/Users/shaileshrajan/Desktop/ml_proj/VAE_project/processed_data/All"
    
    print(f"Using device: {DEVICE}")

    # ---- 1. Load the full dataset to get the complete label map ---- #
    # This ensures we have all categories and their correct integer indices.
    print("Loading dataset to build label map...")
    full_dataset = FashionDataset(IMG_DIR, CSV_PATH)
    label_map = full_dataset.label_map
    num_classes = 4

    # ---- 2. Validate the input category ---- #
    if category_name not in label_map:
        print(f"Error: Category '{category_name}' not found in the dataset.", file=sys.stderr)
        print(f"Available categories are: {list(label_map.keys())}", file=sys.stderr)
        sys.exit(1)

    # ---- 3. Prepare the Model ---- #
    # Initialize model with the total number of classes it was trained on.
    print("Loading pre-trained model...")
    cvae = CVAE(img_channels=3, img_size=IMG_SIZE, latent_dim=LATENT_DIM, num_classes=num_classes).to(DEVICE)
    cvae.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    cvae.eval() # Set model to evaluation mode

    # ---- 4. Create Latent and Label Tensors ---- #
    # Generate random noise from the latent space
    z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(DEVICE)

    # Get the integer index for the desired category
    category_index = label_map[category_name]
    
    # Create a 1D tensor of integer labels (this is what nn.Embedding expects)
    labels = torch.tensor([category_index] * NUM_SAMPLES, dtype=torch.long).to(DEVICE)

    # ---- 5. Generate Images ---- #
    print(f"Generating {NUM_SAMPLES} images for category: '{category_name}'...")
    with torch.no_grad():
        generated_images = cvae.decode(z, labels)

    # ---- 6. Visualize and Save ---- #
    grid = make_grid(generated_images.cpu(), nrow=4, padding=2) # Arrange in a 1x4 grid
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0)) # Convert from (C, H, W) to (H, W, C)
    plt.title(f"Generated Samples for '{category_name}'")
    plt.axis("off")
    
    output_filename = f"generated_{category_name.replace(' ', '_')}.png"
    plt.savefig(output_filename)
    print(f"Saved generated images to '{output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a pre-trained Conditional VAE.")
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help='The subCategory to generate images for (e.g., "Shoes", "Jeans", "Heels").'
    )
    args = parser.parse_args()
    generate_images(args.category)