import os
import numpy as np
from diffusion_main import Diffusion, inverse_data_transform
import torch
import pickle
from models.transformer_film_mono import TransformerDDPME
from singletrack_VAE import singletrack_vae, db_processing
import uuid
import random

current_dir = os.getcwd()
num_samples_to_generate = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_diffusion = 64

categories = {"emotions": 4}
model = TransformerDDPME(categories).to(device)

diffusion = Diffusion(noise_steps=model.num_timesteps, batch_size=batch_size_diffusion, vocab_size=model.vocab_size,
                      time_steps=model.seq_len)
checkpoint = torch.load("./min_checkpoint.pth.tar", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
print("epoch:", checkpoint["epoch"])


nesmdb_shared_library_rel_path = "ext_nseq_nesmdb_single_lib.so"
batch_size = 64
temperature = 0.002
total_steps = 32

model_rel_path = "cat-mel_2bar_big.tar"
model_path = os.path.join(current_dir, model_rel_path)
db_type = "nesmdb_singletrack"
nesmdb_shared_library_rel_path = "ext_nseq_nesmdb_single_lib.so"

model_path = os.path.join(current_dir, model_rel_path)
nesmdb_shared_library_path = os.path.join(current_dir, nesmdb_shared_library_rel_path)

db_proc = db_processing(nesmdb_shared_library_path, db_type)
vae = singletrack_vae(model_path, batch_size)

# sample and decode
num_samples_to_generate = 1
random_emotions = [random.randint(0, 3) for _ in range(num_samples_to_generate)]
random_emotions = torch.tensor(random_emotions).to(device)
sampled_latents = diffusion.sample(model, num_samples_to_generate, random_emotions, cfg_scale=3)
batch_transformed = inverse_data_transform(torch.Tensor.cpu(sampled_latents), -14., 14.)

melodies_decoded = vae.decode_sequence(batch_transformed, total_steps, temperature)
song_data_ = db_proc.song_from_melody(melodies_decoded[0])

midi = db_proc.midi_from_song(song_data_)
midi.save("./sampled_song.mid")

