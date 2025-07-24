import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)
print('-----------------------------')

## visualize images and check for shape
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random

# Path setup
DATA_DIR = "/Users/shaileshrajan/Desktop/ml_proj/VAE_project/Fashion Product Images"
IMG_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "styles.csv")

# Load metadata
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')  # Handle bad rows
print("Total entries in CSV:", len(df))

# Some entries have nulls, drop them
df = df.dropna(subset=["id", "articleType", "subCategory", "masterCategory"])

# Ensure 'id' is treated as str and append .jpg
df["filename"] = df["id"].astype(str) + ".jpg"

# Filter only those whose images exist
df = df[df["filename"].apply(lambda x: os.path.isfile(os.path.join(IMG_DIR, x)))]

print("Cleaned entries with existing images:", len(df))

# Show a few rows
print("\nSample rows:")
print(df.head())

print(df["subCategory"].unique())

# Category counts
# plt.figure(figsize=(10, 5))
# sns.countplot(data=df, x="masterCategory", order=df["masterCategory"].value_counts().index)
# plt.title("Master Category Distribution")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('eda/category_counts.png')

# # Subcategory distribution (top 10)
# plt.figure(figsize=(12, 6))
# top_subcats = df["subCategory"].value_counts().nlargest(10).index
# sns.countplot(data=df[df["subCategory"].isin(top_subcats)], x="subCategory")
# plt.title("Top 10 Subcategories")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('eda/subcategory_distribution.png')

# # Function to display a few images
# def show_images(df_sample):
#     plt.figure(figsize=(12, 8))
#     for i, (_, row) in enumerate(df_sample.iterrows()):
#         img_path = os.path.join(IMG_DIR, row["filename"])
#         try:
#             img = Image.open(img_path).convert("RGB")
#             plt.subplot(2, 5, i + 1)
#             plt.imshow(img)
#             plt.title(row["subCategory"], fontsize=8)
#             plt.axis("off")
#         except:
#             continue
#     plt.tight_layout()
#     plt.savefig(f'eda/sample/{i}')

# # Visualize 10 random images
# sample_df = df.sample(10)
# show_images(sample_df)

# # Check image shapes
# img_shapes = []
# for i in range(20):  # sample 20 images
#     path = os.path.join(IMG_DIR, df.iloc[i]["filename"])
#     with Image.open(path) as img:
#         img_shapes.append(img.size)

# print("\nSample image sizes (width, height):")
# print(img_shapes)
