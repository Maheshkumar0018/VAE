import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchvision
import umap.umap_ as umap
import torch
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import os
# import matplotlib
# matplotlib.use('Agg')

# Define KL Divergence loss for VAE
def kl_divergence_loss(mu, logvar):
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# Encoder Class
class Encoder(nn.Module):
    def __init__(self, im_chan=1, z_dim=8, hidden_dim=16):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(im_chan, hidden_dim, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2 * 7 * 7, z_dim * 2)  # to mean and logvar
        )

    def forward(self, image):
        output = self.encoder(image)
        mean = output[:, :self.z_dim]
        logvar = output[:, self.z_dim:]
        return mean, logvar

# Decoder Class
class Decoder(nn.Module):
    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim, 7, 7)),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, im_chan, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid to ensure output is between 0 and 1
        )

    def forward(self, z):
        return self.decoder(z)

# VAE Class
class VAE(nn.Module):
    def __init__(self, im_chan=1, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = Encoder(im_chan, z_dim)
        self.decoder = Decoder(z_dim, im_chan)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(logvar / 2)
        q = Normal(mu, std)
        z = q.rsample()
        return self.decoder(z), mu, logvar


# Function to display a grid of images
def show_images_grid(images, title=''):
    """Utility function to show images grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(torchvision.utils.make_grid(images, padding=2, normalize=True).cpu().numpy().transpose((1, 2, 0)))
    ax.title.set_text(title)
    plt.show()

# Train the VAE model function
def train_model(train_loader, device, epochs=10, z_dim=16):
    # Initialize the model
    model = VAE(im_chan=1, z_dim=z_dim).to(device)
    model_opt = torch.optim.Adam(model.parameters())
    reconstruction_loss = nn.BCELoss(reduction='sum')

    # Train the model
    for epoch in range(epochs):
        total_loss, total_kl_loss, total_recon_loss = 0, 0, 0
        model.train()
        for images, _ in tqdm(train_loader):
            images = images.to(device)
            model_opt.zero_grad()

            # Forward pass
            recon_images, mu, logvar = model(images)

            # Calculate losses
            R_loss = reconstruction_loss(recon_images, images)
            KLD_loss = kl_divergence_loss(mu, logvar)
            loss = R_loss + KLD_loss

            # Backward and optimize
            loss.backward()
            model_opt.step()

            total_loss += loss.item()
            total_kl_loss += KLD_loss.item()
            total_recon_loss += R_loss.item()

        # Compute average losses
        avg_loss = total_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"KL Div: {avg_kl_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}")

        # Show images after each epoch
        # show_images_grid(images.cpu(), title='Input Images - Last Batch')
        # show_images_grid(recon_images.cpu(), title='Reconstructed Images - Last Batch')

    return model


def visualize_latent_space(model, data_loader, device, method='UMAP', num_samples=10000, save_path='latent_space_plot.png'):
    model.eval()
    latents = []
    labels = []
    total_samples = 0

    with torch.no_grad():
        for data, label in data_loader:
            if total_samples >= num_samples:
                break
            data = data.to(device)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu())
            labels.append(label.cpu())
            total_samples += data.size(0)

    latents = torch.cat(latents, dim=0).numpy()[:num_samples]
    labels = torch.cat(labels, dim=0).numpy()[:num_samples]

    if method == 'TSNE':
        tsne = TSNE(n_components=2, verbose=1)
        tsne_results = tsne.fit_transform(latents)
        x = tsne_results[:, 0]
        y = tsne_results[:, 1]
    elif method == 'UMAP':
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(latents)
        x = embedding[:, 0]
        y = embedding[:, 1]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(x, y, c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title(f'VAE Latent Space with {method}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)

    plt.savefig(save_path)  # Save the figure to a file
    plt.close()


# Set device and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
vae = train_model(train_loader, device, epochs=10, z_dim=8)

# Visualize the latent space
# visualize_latent_space(vae, train_loader, device, method='UMAP', num_samples=10000)

def get_data_predictions(model, data_loader, device):
    model.eval()
    latents_mean = []
    latents_logvar = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            mu, logvar = model.encoder(data)
            latents_mean.append(mu.cpu())
            latents_logvar.append(logvar.cpu())
            labels.append(label.cpu())
    latents_mean = torch.cat(latents_mean, dim=0)
    latents_logvar = torch.cat(latents_logvar, dim=0)
    labels = torch.cat(labels, dim=0)
    return latents_mean, latents_logvar, labels

def get_classes_mean(class_to_idx, labels, latents_mean, latents_logvar):
    classes_mean_std = {}
    for class_name, class_id in class_to_idx.items():
        mask = labels == class_id
        latents_mean_class = latents_mean[mask].mean(dim=0)
        latents_logvar_class = latents_logvar[mask].mean(dim=0)
        classes_mean_std[class_id] = (latents_mean_class, torch.exp(0.5 * latents_logvar_class))
    return classes_mean_std


def show_image(image, title=''):
    plt.figure()
    plt.imshow(image.detach().squeeze(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()



def traverse_two_latent_dimensions(model, mu, std, n_samples=10, dim_1=0, dim_2=1, class_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_size = 28

    # Define the folder for saving images
    image_folder = f'./generated_images/class_{class_id}'
    os.makedirs(image_folder, exist_ok=True)

    # Define the sampling range using standard deviation
    grid_x = torch.linspace(mu[dim_1] - 3*std[dim_1], mu[dim_1] + 3*std[dim_1], n_samples)
    grid_y = torch.linspace(mu[dim_2] - 3*std[dim_2], mu[dim_2] + 3*std[dim_2], n_samples)

    # Generate and save images
    model.eval()
    with torch.no_grad():
        for xi in range(n_samples):
            for yi in range(n_samples):
                z_sample = mu.clone().detach()
                z_sample[dim_1] = grid_x[xi]
                z_sample[dim_2] = grid_y[yi]

                # Decode the latent sample
                recon_image = model.decoder(z_sample.unsqueeze(0)).squeeze().cpu().numpy()

                # Print debug information
                print(f"Generating image for class {class_id} at grid position x{xi} y{yi}: dim_1={z_sample[dim_1]}, dim_2={z_sample[dim_2]}")

                # Plot and save each image
                plt.figure(figsize=(2.8, 2.8))
                plt.imshow(recon_image, cmap='gray')
                plt.title(f'Class {class_id} x{xi} y{yi}')
                plt.axis('off')
                plt.savefig(f'{image_folder}/digit_{xi}_{yi}.png')
                plt.close()

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latents_mean, latents_logvar, labels = get_data_predictions(vae, train_loader, device)
classes_mean_std = get_classes_mean(train_loader.dataset.class_to_idx, labels, latents_mean, latents_logvar)

for class_id in range(10):
    mu, std = classes_mean_std[class_id]
    traverse_two_latent_dimensions(vae, mu, std, n_samples=8, dim_1=3, dim_2=4, class_id=class_id)