import pandas as pd
import os
from tqdm import tqdm

# --- Configuration ---
CSV_PATH = '/Users/shaileshrajan/Desktop/ml_proj/VAE_project/Fashion Product Images/styles.csv'
IMAGE_DIR = '/Users/shaileshrajan/Desktop/ml_proj/VAE_project/processed_data/All'
CLEANED_CSV_PATH = '/Users/shaileshrajan/Desktop/ml_proj/VAE_project/styles_cleaned.csv' # Output file

# --- Script ---
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
print(f"Original number of entries: {len(df)}")

def image_exists(image_id):
    """Checks if an image file exists for a given ID."""
    # Ensure image_id is a string before joining
    path = os.path.join(IMAGE_DIR, str(image_id) + ".jpg")
    return os.path.exists(path)

# Use tqdm for a progress bar while checking files
tqdm.pandas(desc="Checking for missing images")
keep_mask = df['id'].progress_apply(image_exists)

# Filter the dataframe to keep only rows with existing images
cleaned_df = df[keep_mask]
print(f"Number of entries after cleaning: {len(cleaned_df)}")

# Save the cleaned dataframe to a new CSV file
cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
print(f"Cleaned CSV saved to: {CLEANED_CSV_PATH}")