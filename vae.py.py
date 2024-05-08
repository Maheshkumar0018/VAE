import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters-
num_epochs = 50
batch_size = 64
learning_rate = 0.001

# Define transformations for CelebA dataset-
transforms_apply = transforms.Compose(
    [
        torchvision.transforms.CenterCrop((128, 128)),
        transforms.ToTensor()
    ]
)


class UnlabeledImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Path to the directory containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image



# Paths to your datasets
train_dir = './inference/train/'
test_dir = './inference/test/'
val_dir = './inference/val/'

# Create dataset objects
train_dataset = UnlabeledImageDataset(directory=train_dir, transform=transforms_apply)
test_dataset = UnlabeledImageDataset(directory=test_dir, transform=transforms_apply)
val_dataset = UnlabeledImageDataset(directory=val_dir, transform=transforms_apply)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
valid_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


print("DataLoaders was Created !!!")

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :128, :128]
    

class VAE(nn.Module):
    def __init__(self, latent_space = 200):
        super(VAE, self).__init__()
        
        self.latent_space = latent_space
        
        # Define encoder architecture-
        self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels = 3, out_channels = 32,
                    stride = 2, kernel_size = 3,
                    bias = False, padding = 1
                ),
                nn.BatchNorm2d(num_features = 32),
                nn.LeakyReLU(0.1, inplace = True),
                nn.Dropout2d(p = 0.25),
                
                nn.Conv2d(
                    in_channels = 32, out_channels = 64,
                    stride = 2, kernel_size = 3,
                    bias = False, padding = 1
                ),
                nn.BatchNorm2d(num_features = 64),
                nn.LeakyReLU(0.1, inplace = True),
                nn.Dropout2d(p = 0.25),
                
                nn.Conv2d(
                    in_channels = 64, out_channels = 64,
                    stride = 2, kernel_size = 3,
                    bias = False, padding = 1
                ),
                nn.BatchNorm2d(num_features = 64),
                nn.LeakyReLU(0.1, inplace = True),
                nn.Dropout2d(p = 0.25),
                
                nn.Conv2d(
                    in_channels = 64, out_channels = 64,
                    stride = 2, kernel_size = 3,
                    bias = False, padding = 1
                ),
                nn.BatchNorm2d(num_features = 64),
                nn.LeakyReLU(0.1, inplace = True),
                nn.Dropout2d(p = 0.25),
                nn.Flatten(),
        )    
        
        # Define mean & log-variance vectors to represent latent space 'z'-
        self.mu = torch.nn.Linear(in_features = 4096, out_features = self.latent_space)
        self.log_var = torch.nn.Linear(in_features = 4096, out_features = self.latent_space)
        
        # Define encoder architecture-
        self.decoder = nn.Sequential(
                torch.nn.Linear(
                    in_features = self.latent_space, out_features = 4096
                ),
                Reshape(-1, 64, 8, 8),
                
                nn.ConvTranspose2d(
                    in_channels = 64, out_channels = 64,
                    stride = 2, kernel_size = 3
                ),
                nn.BatchNorm2d(num_features = 64),
                nn.LeakyReLU(0.1, inplace = True),
                nn.Dropout2d(p = 0.25),
                
                nn.ConvTranspose2d(
                    in_channels = 64, out_channels = 64,
                    stride = 2, kernel_size = 3,
                    padding = 1
                ),
                nn.BatchNorm2d(num_features = 64),
                nn.LeakyReLU(0.1, inplace = True),
                nn.Dropout2d(p = 0.25),
                
                nn.ConvTranspose2d(
                    in_channels = 64, out_channels = 32,
                    stride = 2, kernel_size = 3,
                    padding = 1),
                nn.BatchNorm2d(num_features = 32),
                nn.LeakyReLU(0.1, inplace = True),
                nn.Dropout2d(p = 0.25),
                
                nn.ConvTranspose2d(
                    in_channels = 32, out_channels = 3,
                    stride = 2, kernel_size = 3,
                    padding = 1
                ),
                
                # Trim: 3x129x129 -> 3x128x128
                Trim(),
            
                # Due to input image being in the scale [0, 1], use sigmoid-
                nn.Sigmoid()
                )

        
    # def reparameterize(self, mu, log_var):
    #     # 'eps' samples from a normal standard distribution to add
    #     # stochasticity to the sampling process-
    #     # eps = torch.randn_like(log_var).to(log_var.get_device())
    #     eps = torch.randn(mu.size(0), mu.size(1)).to(mu.get_device())
    #     z = mu + eps * torch.exp(log_var / 2.0) 
    #     return z
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        Ensures that samples are on the same device as mu and log_var.
        """
        # Create the standard normal distribution 'eps' directly on the device of 'mu'.
        eps = torch.randn_like(mu)  # This ensures that eps is on the same device as mu
        z = mu + eps * torch.exp(log_var / 2.0)
        return z
    
    
    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.mu(x), self.log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
    
        
    def forward(self, x):
        # Encode input data-
        x = self.encoder(x)
        ##NOTE: The line of code above does NOT give us the latent vector!
        
        # Get mean & log-var vectors representing latent space distribution-
        mu, log_var = self.mu(x), self.log_var(x)
        
        # Obtain the latent vector 'z' using reparameterization-
        z = self.reparameterize(mu = mu, log_var = log_var)
        
        # Get reconstruction using 'z' as input to decoder-
        x_recon = self.decoder(z)
        
        return x_recon, mu, log_var
        
        
    def shape_computation(self, x):
        print(f"Input shape: {x.shape}")
        x = self.encoder(x)
        print(f"Encoder output shape: {x.shape}")
        mu, log_var = self.mu(x), self.log_var(x)
        z = self.reparameterize(mu = mu, log_var = log_var)
        print(f"mu.shape: {mu.shape}, log_var.shape: {log_var.shape} &"
              f" z.shape: {z.shape}")
        
        x_recon = self.decoder(z)
        print(f"Decoder output shape: {x_recon.shape}")
        del x, x_recon, mu, log_var, z
        return None


def total_loss(data, data_recon, mu, log_var, alpha = 1):
    '''
    Function to compute loss = reconstruction loss * reconstruction_term_weight + KL-Divergence loss.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    Inputs:
    1. mu: mean from the latent vector
    2. logvar: log variance from the latent vector
    3. alpha (int): Hyperparameter to control the importance of reconstruction
    loss vs KL-Divergence Loss - reconstruction term weight
    4. data: training data
    5. data_recon: VAE's reconstructed data
    '''
    
    # Compute KL-Divergence loss:
    
    # Sum over latent dimensions-
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), axis = 1)
    # kl_div = -0.5 * torch.sum(1 + log_var - (mu **2) - torch.exp(log_var), axis = 1)
    
    '''
    Omitting 'axis' will give bad results as it will sum over everything!
    First, sum over the latent dimensions and then average over the batches.
    '''
    
    # kl_div.shape
    # torch.Size([64])
    
    batchsize = kl_div.size(0)

    # Compute average KL-divergence over batch size-
    kl_div = kl_div.mean()
    
    
    # Compute Reconstruction loss:
    
    reconstruction_loss_fn = F.mse_loss
    recon_loss = reconstruction_loss_fn(data_recon, data, reduction = 'none')
    
    # recon_loss.shape
    # torch.Size([32, 1, 28, 28])
    
    # Sum over all pixels-
    # Reshape recon_loss so that it is the batchsize and a vector. So, instead
    # of having a tensor, it is now a matrix (table). Then, sum over the pixels.
    # This is equivalent to summing over the latent dimensions for kl_div above.
    # We are summing first the squared error over the pixels and then average over
    # the batch dimensions below-
    recon_loss = recon_loss.view(batchsize, -1).sum(axis = 1)
    
    # recon_loss.shape
    # torch.Size([64, 784])
    
    # Average over mini-batch dimension-
    recon_loss = recon_loss.mean()
    
    final_loss = (alpha * recon_loss) + kl_div
    
    return final_loss, recon_loss, kl_div


def train_one_epoch(model, dataloader, alpha):
    
    # Push model to 'device'-
    model.to(device)
    
    # Enable training mode-
    model.train()
    
    # Initialize variables to keep track of 3 losses-
    running_final_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    
    
    with tqdm(train_loader, unit = 'batch') as tepoch:
        for images in tepoch:
            tepoch.set_description(f"Training: ")
            # print(images.shape, labels.shape)
            
            # Push images to 'device'-
            images = images.to(device)
            
            # Empty accumulated gradients-
            optimizer.zero_grad()
            
            # Perform forward propagation-
            recon_images, mu, log_var = model(images)
            
            # Compute different losses-
            final_loss, recon_loss, kl_div_loss = total_loss(
                data = images, data_recon = recon_images,
                mu = mu, log_var = log_var,
                alpha = alpha
            )
            
            # Update losses-
            running_final_loss += final_loss.item()
            running_kl_loss += kl_div_loss.cpu().detach().numpy()
            running_recon_loss += recon_loss.cpu().detach().numpy()
            
            # Compute gradients wrt total loss-
            final_loss.backward()
            
            # Perform gradient descent-
            optimizer.step()
    
    # Compute losses as float values-
    train_loss = running_final_loss / len(dataloader.dataset)
    kl_loss = running_kl_loss / len(dataloader.dataset)
    recon_loss = running_recon_loss / len(dataloader.dataset)
    
    return train_loss, kl_loss, recon_loss
        
def validate_one_epoch(model, dataloader, alpha):
    
    # Place model to device-
    model.to(device)
    
    # Enable evaluation mode-
    model.eval()
    
    running_final_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    
    with torch.no_grad():
        with tqdm(test_loader, unit = 'batch') as tepoch:
            for images in tepoch:
                tepoch.set_description(f"Validation: ")
                
                # Push data points to 'device'-
                images = images.to(device)
                
                # Perform forward propagation-
                recon_images, mu, log_var = model(images)
                
                # Compute different losses-
                final_loss, recon_loss, kl_div_loss = total_loss(
                    data = images, data_recon = recon_images,
                    mu = mu, log_var = log_var,
                    alpha = alpha
                )
                
                # Update losses-
                running_final_loss += final_loss.item()
                running_kl_loss += kl_div_loss.cpu().detach().numpy()
                running_recon_loss += recon_loss.cpu().detach().numpy()
            
    val_loss = running_final_loss / len(dataloader.dataset)
    val_kl_loss = running_kl_loss / len(dataloader.dataset)
    val_recon_loss = running_recon_loss / len(dataloader.dataset)
    
    return val_loss, val_kl_loss, val_recon_loss


model = VAE(latent_space = 200)
model.to(device)
print(model)


# Count number of layer-wise parameters and total parameters-
tot_params = 0
for param in model.parameters():
    #print(f"layer.shape = {param.shape} has {param.nelement()} parameters")
    tot_params += param.nelement()
print(f"Total number of parameters in VAE model = {tot_params}")

# Before training the VAE neural network model, there should be some _initial variance_.
for x in model.mu.parameters():
    print(x.shape, x.nelement())


optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  


# Initialize parameters for Early Stopping manual implementation-
best_val_loss = 100
# Python3 dict to contain training metrics-
train_history = {}
# Specify alpha - Hyperparameter to control the importance of reconstruction
# loss vs KL-Divergence Loss-
alpha = 1

print("Set for Training !!!")

for epoch in range(1, num_epochs + 1):
    '''
    # Manual early stopping implementation-
    if loc_patience >= patience:
        print("\n'EarlyStopping' called!\n")
        break
    '''
    
    # Train model for 1 epoch-
    train_loss, kl_train_loss, recon_train_loss = train_one_epoch(
        model = model, dataloader = train_loader,
        alpha = alpha
    )
    
    # Get validation after 1 epoch-
    val_loss, val_kl_loss, val_recon_loss = validate_one_epoch(
        model = model, dataloader = valid_loader,
        alpha = alpha
    )
    
    # Store model performance metrics in Python3 dict-
    train_history[epoch] = {
        'train_loss': train_loss,
        'train_recon_loss': kl_train_loss,
        'train_kl_loss': kl_train_loss,
        'val_loss': val_loss,
        'val_recon_loss': val_recon_loss,
        'val_kl_loss': val_kl_loss
    }
    
    print(f"Epoch = {epoch}; train loss = {train_loss:.4f}",
          f", kl-loss = {kl_train_loss:.4f}, recon loss = {recon_train_loss:.4f}",
          f", val loss = {val_loss:.4f}, val kl-loss = {val_kl_loss:.4f}",
          f" & val recon loss = {val_recon_loss:.4f}"
         )
    
    
    # Save 'best' model so far-
    if val_loss < best_val_loss:
        # Update for lowest val_loss so far-
        best_val_loss = val_loss
        
        print(f"Saving model with lowest val_loss = {val_loss:.4f}\n")
        
        # Save trained model with 'best' validation loss-
        torch.save(model.state_dict(), "VAE_best_val_loss.pth")
    
    
    '''
    # Code for manual Early Stopping:
    if (val_epoch_loss < best_val_loss) and \
    (np.abs(val_epoch_loss - best_val_loss) >= minimum_delta):

        # update 'best_val_loss' variable to lowest loss encountered so far-
        best_val_loss = val_loss
        
        # reset 'loc_patience' variable-
        loc_patience = 0

        print(f"Saving model with lowest val_loss = {val_loss:.4f}\n")
        
        # Save trained model with 'best' validation accuracy-
        torch.save(model.state_dict(), "VAE_LeNet5_MNIST_best_model.pth")
        
    else:  # there is no improvement in monitored metric 'val_loss'
        loc_patience += 1  # number of epochs without any improvement
    '''
    

# Save trained model at last epoch-
torch.save(model.state_dict(), "./VAE_last_epoch.pth")

# Save training metrics as pickled Python3 dict-
with open("VAE_training_metrics.pkl", "wb") as file:
    pickle.dump(train_history, file)

# Load trained metrics-
with open("./VAE_training_metrics.pkl", "rb") as file:
    train_history = pickle.load(file)

plt.figure(figsize = (9, 7))
plt.plot([train_history[e]['train_loss'] for e in train_history.keys()], label = 'train_loss')
plt.plot([train_history[e]['val_loss'] for e in train_history.keys()], label = 'val_loss')
plt.title("VAE Training Visualization: Total Loss")
plt.legend(loc = 'best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

plt.figure(figsize = (9, 7))
plt.plot([train_history[e]['train_recon_loss'] for e in train_history.keys()], label = 'train_recon_loss')
plt.title("VAE Training Visualization: Training Reconstruction Loss")
plt.legend(loc = 'best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

plt.figure(figsize = (9, 7))
plt.plot([train_history[e]['val_recon_loss'] for e in train_history.keys()], label = 'val_recon_loss')
plt.title("VAE Training Visualization: Validation Reconstruction Loss")
plt.legend(loc = 'best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

plt.figure(figsize = (9, 7))
plt.plot([train_history[e]['train_kl_loss'] for e in train_history.keys()], label = 'train_kl_loss')
plt.title("VAE Training Visualization: KL-Divergence loss")
plt.legend(loc = 'best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

plt.figure(figsize = (9, 7))
plt.plot([train_history[e]['val_kl_loss'] for e in train_history.keys()], label = 'val_kl_loss')
plt.title("VAE Training Visualization: KL-Divergence loss")
plt.legend(loc = 'best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# Initialize and load trained weights from before-
trained_model = VAE(latent_space = 200).to(device)
trained_model.load_state_dict(torch.load("./VAE_last_epoch.pth"))

val_loss, val_kl_loss, val_recon_loss = validate_one_epoch(
    model = model, dataloader = valid_loader,
    alpha = 1
)

print(f"Trained model validation metrics: Total loss = {val_loss:.4f}"
      f", KL-div loss = {val_kl_loss:.4f} & reconstruction loss = "
      f"{val_recon_loss:.4f}"
     )

images = next(iter(train_loader))
images = images.to(device)
recon_images, mu, log_var = trained_model(images)

recon_images = recon_images.cpu().detach().numpy()
images = images.cpu().detach().numpy()

recon_images = np.transpose(recon_images, (0, 2, 3, 1))
images = np.transpose(images, (0, 2, 3, 1))

# Visualize 30 images from training set-
plt.figure(figsize = (15, 13))
for i in range(30):
    # 6 rows & 5 columns-
    plt.subplot(6, 5, i + 1)
    plt.imshow(images[i])
    
plt.suptitle("Sample training images")
plt.show()
plt.savefig('Sample_training_images.png')


# Visualize 30 reconstructed images-
plt.figure(figsize = (15, 13))
for i in range(30):
    # 6 rows & 5 columns-
    plt.subplot(6, 5, i + 1)
    plt.imshow(recon_images[i])
    
plt.suptitle("Sample Reconstructed images")
plt.show()
plt.savefig('Sample_Reconstruced_images.png')