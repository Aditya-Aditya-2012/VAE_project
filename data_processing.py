import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Settings
ORIG_DATA_DIR = "/Users/shaileshrajan/Desktop/ml_proj/VAE_project/Fashion Product Images/images"
CSV_PATH = "/Users/shaileshrajan/Desktop/ml_proj/VAE_project/Fashion Product Images/styles.csv"
OUTPUT_DIR = "processed_data/All"  # All images under one folder
TARGET_SIZE = (64, 64)  # Resize all images to 64x64 for VAE

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
df = df.dropna(subset=["id"])
df["filename"] = df["id"].astype(str) + ".jpg"

# Filter only files that exist
df = df[df["filename"].apply(lambda x: os.path.isfile(os.path.join(ORIG_DATA_DIR, x)))]

print(f"Found {len(df)} valid image entries.")

# Resize and save
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    img_path = os.path.join(ORIG_DATA_DIR, row["filename"])
    out_path = os.path.join(OUTPUT_DIR, row["filename"])
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(TARGET_SIZE)
        img.save(out_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
