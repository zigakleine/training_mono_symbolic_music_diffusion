from nesmdb_dataset import NesmdbMidiDataset
import numpy as np
is_reduced = False
from torch.utils.data import DataLoader
import torch

def normalize_dataset(batch, data_min, data_max, std_dev_masks):
    """Normalize dataset to range [-1, 1]."""
    batch = (batch - data_min) / (data_max - data_min)
    batch = 2. * batch - 1.

    if is_reduced:
        enc_tracks = np.split(batch, 4, axis=0)
        enc_tracks_reduced = []
        for enc_track, std_dev_mask in zip(enc_tracks, std_dev_masks):

            enc_track_reduced = enc_track[:, std_dev_mask]
            enc_tracks_reduced.append(enc_track_reduced)

        enc_tracks_reduced = np.vstack(enc_tracks_reduced)
    else:
        enc_tracks_reduced = batch

    return enc_tracks_reduced


std_devs_masks = None
dmin=-14.
dmax=14.
batch_size = 64

dataset = NesmdbMidiDataset(transform=normalize_dataset, std_dev_masks=std_devs_masks, dmin=dmin, dmax=dmax)
train_ds, test_ds = torch.utils.data.random_split(dataset, [100127, 3097])

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)

for step, (batch, l) in enumerate(train_loader):
    print(batch.numpy())