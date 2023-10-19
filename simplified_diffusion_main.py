from diffusion_main import Diffusion
import tqdm
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import pickle
from models.transformer_film_mono import TransformerDDPME



def normalize_dataset(batch, data_min, data_max):
    """Normalize dataset to range [-1, 1]."""
    batch = (batch - data_min) / (data_max - data_min)
    batch = 2. * batch - 1.

    enc_tracks_reduced = batch

    return enc_tracks_reduced

def inverse_data_transform(batch, data_min, data_max):

    batch = (batch + 1.) / 2.
    batch = (data_max - data_min) * batch + data_min
    batch = batch.numpy()
    batch_ = []
    for enc_tracks in batch:

        enc_tracks_split = np.split(enc_tracks, 4, axis=1)
        enc_tracks_reconstructed = enc_tracks_split
        enc_tracks_reconstructed = np.vstack(enc_tracks_reconstructed)
        batch_.append(enc_tracks_reconstructed)

    return np.array(batch_)

def setup_logging(run_name, current_dir):

    os.makedirs(os.path.join(current_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name, "generated"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name, "graphs"), exist_ok=True)


dmin = -14.
dmax = 14.
epochs_num = 250010
lr = 3e-5
batch_size = 1
current_dir = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_type = "song_red"
run_name = "mono_overfit_test_2"

categories = {"emotions": 4}

model = TransformerDDPME(categories).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.98)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5, patience=700)
mse = nn.MSELoss()

setup_logging(run_name, current_dir)

diffusion = Diffusion(noise_steps=model.num_timesteps, batch_size=batch_size, vocab_size=model.vocab_size,
                      time_steps=model.seq_len)


train_losses = []

current_dir = os.getcwd()
# to_save_dir = "/storage/local/ssd/zigakleine-workspace"
to_save_dir = os.getcwd()

songs_path = "./training_song_mono.pkl"
songs = pickle.load(open(songs_path, "rb"))
song = songs[0]


song = song[None, :, :]
song = normalize_dataset(torch.tensor(song), dmin, dmax)

for epoch in range(epochs_num):

    logging.info(f"Starting epoch{epoch}:")

    train_count = 0
    train_loss_sum = 0

    emotions = None

    batch = song.to(device)

    t = diffusion.sample_timesteps(1).to(device)

    x_t, noise = diffusion.noise_latents(batch, t)
    predicted_noise = model(x_t, t, emotions)

    loss = mse(noise, predicted_noise)
    train_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()
    scheduler.step(train_loss)

    train_losses.append(train_loss)
    current_lr = optimizer.param_groups[0]['lr']
    logging.info(f"Learning rate at epoch  epoch:{current_lr}")
    logging.info(f"Epoch {epoch} mean training loss: {train_loss}")

    if epoch % 1000 == 0:
        epochs = range(len(train_losses))
        plt.plot(epochs, train_losses, 'r', label='Training Loss')
        # Add labels and a legend
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation and Training Losses')
        plt.legend()
        loss_plot_abs_path = os.path.join(to_save_dir, "results", run_name, "graphs", f"loss_plot_{epoch}.png")
        plt.savefig(loss_plot_abs_path)
        plt.clf()

    if epoch % 1000 == 0:

        sampled_latents = diffusion.sample(model, 1, None, cfg_scale=0)

        batch_transformed = inverse_data_transform(torch.Tensor.cpu(sampled_latents), dmin, dmax)
        generated_batch_abs_path = os.path.join(to_save_dir, "results", run_name, "generated", f"{epoch}_epoch_batch.pkl")
        file = open(generated_batch_abs_path, 'wb')
        pickle.dump(batch_transformed, file)
        file.close()
