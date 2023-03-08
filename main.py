from torch.optim import Adam
import math
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from stanford_cars.dataset import car_dataloader, tensor_to_img
from src.device import device
from tqdm import tqdm


# ===== Step 1 =====


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    # same as .reshape( (batch_size,) + ((1,) * (len(x_shape) - 1)) )
    result = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    return result


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    # t will be of the form [1, 126, 3, 6, 256, ...], a random list of timestamp
    # sqrt_alphas_cumprod_t will be like a slice sqrt_alphas_cumprod[[1, 126, 3, 6, 256, ...]]
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    # so is sqrt_one_minus_alphas_cumprod_t
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    # x_t = \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\cdot z_t
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
IMG_SIZE = 64
TIMESTEPS_BATCH_SIZE = 128
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def show_tensor_image(image):
    image = tensor_to_img(image)
    plt.imshow(image)


# ===== Step 2 =====
"""
For a great introduction to UNets, have a look at this post: https://amaarora.github.io/2020/09/13/unet.html.

Key Takeaways:

- We use a simple form of a UNet for to predict the noise in the image

- The input is a noisy image, the ouput the noise in the image

- Because the parameters are shared accross time, we need to tell the network in which timestep we are

- The Timestep is encoded by the transformer Sinusoidal Embedding
We output one single value (mean), because the variance is fixed
"""


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            # since stride = 1, this will keep spacial dimension unchanged
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            # the following will double the spacial dimension
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            # since stride = 1, this will keep spacial dimension unchanged
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            # the following will half the spacial dimension
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # dim = 32
        self.dim = dim

    def forward(self, times):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # a_i = 1/10000^(i/half_dim)
        # embeddings above = [a_1, a_2, a_3, ..., a_16]
        embeddings = times[:, None] * embeddings[None, :]
        # embeddings above <=>
        # t |-> ( sin t*a_1, cos t*a_1, sin t*a_2, cos t*a_2, sin t*a_3, cos t*a_3, ... )
        # for each of 128 t's, therefore the final dimension will be (128, 32)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        ).to(device)

        # Initial projection
        # stride = 1, padding = 1, no change in spatial dimension
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1).to(device)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1],
                                    time_emb_dim).to(device)
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1],
                                        time_emb_dim, up=True).to(device)
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim).to(device)

    def forward(self, x, times):
        # Embedd time
        t = self.time_mlp(times)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            # for the bottom block the x adds an identical copy of x (just poped out) for unity of coding.
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


def get_loss(model, x_0, times):
    # times is of shape (128, )
    x_noisy, noise = forward_diffusion_sample(x_0, times, device)
    # 128 time times, therefore 128 images, x_noisy is of shape [128, 3, 64, 64]
    noise_pred = model(x_noisy, times)

    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, img_path):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        # just create a tensor t of shape (1,), the result is [1], [2], ..., etc
        times = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, times)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i // stepsize + 1)
            show_tensor_image(img.detach().cpu())

    plt.savefig(img_path)


if __name__ == "__main__":
    model = SimpleUnet()
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 100  # Try more!

    for epoch in range(epochs):
        for step, batch in tqdm(enumerate(car_dataloader)):
            optimizer.zero_grad()
            if batch.shape[1] != 3:
                print("The image does not have 3 channels, skip it")
                continue
            t = torch.randint(0, T, (TIMESTEPS_BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t)
            loss.backward()
            optimizer.step()

            if step % 100 == 00:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                img_path = "results" + "/epoch_" + f"{epoch}".zfill(4) + "_step_" + f"{step}".zfill(4) + ".png"
                sample_plot_image(model, img_path)
