import os
import torch
from torch.utils.data import Dataset
import pickle
import json
import numpy as np


class LakhMidiDataset(Dataset):

    def __init__(self, transform=None, dmin=-14., dmax=14.):
        self.subdirectories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
        self.metadata_folder = "db_metadata"
        self.database_folder = "lakh"
        self.current_dir = os.getcwd()
        self.encoded_dir = "/storage/local/ssd/zigakleine-workspace"
        # self.encoded_dir = os.getcwd()
        self.all_lakh_metadata = []
        self.transform = transform

        self.dmin = dmin
        self.dmax = dmax

        for subdir_name in self.subdirectories:
            current_metadata_filename = "lakh_2908_" + subdir_name + ".pkl"
            current_lakh_metadata_abs_path = os.path.join(self.current_dir, self.metadata_folder, self.database_folder,
                                                          current_metadata_filename)

            metadata = pickle.load(open(current_lakh_metadata_abs_path, "rb"))
            for song in metadata:
                if metadata[song]["encodable"]:
                    for i in range(metadata[song]["num_sequences"]):
                        sequence = {"url": metadata[song]["encoded_song_path"], "index": i}
                        self.all_lakh_metadata.append(sequence)

    def __getitem__(self, index):
        enc_seq_rel_path = self.all_lakh_metadata[index]["url"]
        enc_seq_abs_path = os.path.join(self.encoded_dir, enc_seq_rel_path)

        enc_seq = pickle.load(open(enc_seq_abs_path, "rb"))
        enc_seq = enc_seq[self.all_lakh_metadata[index]["index"]]

        if self.transform:
            enc_seq = self.transform(enc_seq, -14., 14.)

        return enc_seq, -1

    def __len__(self):
        return len(self.all_lakh_metadata)
