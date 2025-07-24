import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FashionDataset(Dataset):
    """
    Custom dataset for fashion images.
    It now accepts a 'target_size' tuple to dynamically resize images.
    """
    def __init__(self, image_dir, csv_file, categories=None, target_size=(64, 64), transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            csv_file (str): Path to the csv file with annotations.
            categories (list, optional): List of subCategories to filter by.
            target_size (tuple, optional): The target size (height, width) to resize images to.
            transform (callable, optional): Optional transform to be applied on a sample.
                                            If provided, it overrides the default resize and ToTensor.
        """
        self.image_dir = image_dir
        # Load and filter the dataframe
        self.styles = pd.read_csv(csv_file, on_bad_lines='skip')

        if categories is not None:
            self.styles = self.styles[self.styles['subCategory'].isin(categories)]

        self.styles = self.styles.dropna(subset=['id', 'subCategory'])

        # Get image paths and labels
        self.image_paths = self.styles['id'].astype(str).values
        self.labels = self.styles['subCategory'].values

        # Create label mapping from string labels to integer indices
        unique_labels = sorted(list(set(self.labels)))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_map = {idx: label for label, idx in self.label_map.items()}
        self.numeric_labels = [self.label_map[label] for label in self.labels]

        # 1. Check if a custom transform is provided.
        #    If not, create a default transform using the 'target_size' parameter.
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Construct the full image path (assuming .jpg extension)
        img_name = os.path.join(self.image_dir, self.image_paths[idx] + ".jpg")
        image = Image.open(img_name).convert('RGB')
        label = self.numeric_labels[idx]

        # Apply the transformation
        image = self.transform(image)
        
        return image, label