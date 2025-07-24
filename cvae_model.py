import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, img_channels=3, img_size=64, latent_dim=64, num_classes=45, embedding_dim=10): # <--- MODIFIED
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        # --- MODIFIED: Add an embedding layer for the labels ---
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
        )

        self.flatten_dim = 256 * (img_size // 16) * (img_size // 16)
        
        # --- MODIFIED: Adjust linear layer input size for the embedding dimension ---
        self.fc_mu = nn.Linear(self.flatten_dim + embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + embedding_dim, latent_dim)

        # Decoder
        # --- MODIFIED: Adjust linear layer input size for the embedding dimension ---
        self.decoder_input = nn.Linear(latent_dim + embedding_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x, y):
        batch_size = x.size(0)
        x = self.encoder(x).view(batch_size, -1)
        
        # --- MODIFIED: Convert labels to embeddings before concatenation ---
        y_emb = self.label_embedding(y)
        x = torch.cat([x, y_emb], dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z, y):
        # --- MODIFIED: Convert labels to embeddings before concatenation ---
        y_emb = self.label_embedding(y)
        z = torch.cat([z, y_emb], dim=1)

        x = self.decoder_input(z)
        x = x.view(-1, 256, self.img_size // 16, self.img_size // 16)
        return self.decoder(x)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

def cvae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div