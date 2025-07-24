import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import argparse  # Import argparse
from cvae_model import CVAE, cvae_loss
from fashion_dataset import FashionDataset
from tqdm import tqdm

def main(category_name):
    """
    Trains a CVAE model on a single, specified fashion category.
    
    Args:
        category_name (str): The name of the category to train on (e.g., "Shoes").
    """
    # ---- Dynamic Directory and File Naming ----
    SAVE_DIR = "checkpoints"
    SAMPLES_DIR = "samples"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    # Create a unique checkpoint name for the specified category
    CHECKPOINT_NAME = f"best_model_{category_name}.pth"
    best_loss = float("inf")

    # ---- Configuration ----
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 25
    LATENT_DIM = 64
    LR = 1e-3
    IMG_DIR = "/Users/shaileshrajan/Desktop/ml_proj/VAE_project/processed_data/All"
    CSV_PATH = "/Users/shaileshrajan/Desktop/ml_proj/VAE_project/styles_cleaned.csv"
    
    print(f"--- Training model for category: '{category_name}' on device: {device} ---")

    # ---- Load Dataset for the specific category ----
    # The dataset will now only contain images from the single category provided.
    dataset = FashionDataset(IMG_DIR, CSV_PATH, target_size=(64, 64), categories=[category_name])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # The label_map in the dataset will be {"category_name": 0}, so all labels will be 0.
    # Therefore, the number of classes for the model is 1.
    num_classes_for_model = 1

    # ---- Initialize Model ----
    cvae = CVAE(img_channels=3, img_size=64, latent_dim=LATENT_DIM, num_classes=num_classes_for_model).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=LR)

    # ---- Training Loop ----
    for epoch in range(1, EPOCHS + 1):
        cvae.train()
        running_loss = 0.0

        # Use tqdm for a progress bar over the dataloader
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch"):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            recon_imgs, mu, logvar = cvae(imgs, labels)
            loss = cvae_loss(recon_imgs, imgs, mu, logvar)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

        # ---- Save Best Model ----
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(cvae.state_dict(), os.path.join(SAVE_DIR, CHECKPOINT_NAME))
            print(f"Saved new best model for '{category_name}' at epoch {epoch} with loss {avg_loss:.4f}")

        # ---- Save Sample Reconstructions ----
        cvae.eval()
        with torch.no_grad():
            # Get the first 8 images from the last batch to show reconstruction
            sample_imgs = imgs[:8]
            sample_labels = labels[:8]
            recon_samples, _, _ = cvae(sample_imgs, sample_labels)
            
            # Concatenate original and reconstructed images for comparison
            comparison = torch.cat([sample_imgs, recon_samples])
            
            # Create a unique filename for the sample image
            sample_filename = f"epoch_{epoch}_{category_name}.png"
            save_image(comparison.cpu(), os.path.join(SAMPLES_DIR, sample_filename), nrow=8)

if __name__ == "__main__":
    # ---- Set up Command-Line Argument Parsing ----
    parser = argparse.ArgumentParser(description="Train a Conditional VAE on a specific fashion category.")
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="The single category to train the model on (e.g., 'Shoes', 'Topwear', 'Jeans')."
    )
    args = parser.parse_args()
    
    # Run the main training function with the provided category
    main(args.category)