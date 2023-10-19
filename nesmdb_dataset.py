import os
import torch
from torch.utils.data import Dataset
import pickle
import json
import numpy as np


class NesmdbMidiDataset(Dataset):

    def __init__(self, transform=None, dmin=-14., dmax=14.):

        self.emotions = {
            "Q1": 0,
            "Q2": 1,
            "Q3": 2,
            "Q4": 3
        }

        self.transform = transform

        self.metadata_folder = "db_metadata"
        self.database_folder = "nesmdb"
        self.current_dir = os.getcwd()
        self.encoded_dir = "/storage/local/ssd/zigakleine-workspace"
        # self.encoded_dir = os.getcwd()
        self.all_nesmdb_metadata = []
        self.metadata_filename = "nesmdb_updated2808.pkl"

        self.dmin = dmin
        self.dmax = dmax

        nesmdb_metadata_abs_path = os.path.join(self.current_dir, self.metadata_folder, self.database_folder,
                                                self.metadata_filename)
        metadata = pickle.load(open(nesmdb_metadata_abs_path, "rb"))
        # sequences_num = 0
        for game in metadata:
            for song in metadata[game]["songs"]:
                if song["is_encodable"]:

                    emotion_q = self.emotions[song["emotion_pred_same_vel"]]
                    song_rel_urls = song["encoded_song_urls"]
                    for song_rel_url in song_rel_urls:
                        # if song_rel_url == "nesmdb_encoded/322_SuperMarioBros_/0*+0*p1-p2-tr-no.pkl":

                        # if song_rel_url[-6:] == "p1.pkl":
                        #     emotion_q = self.emotions[song["emotion_pred_p1"]]
                        # elif song_rel_url[-6:] == "p2.pkl":
                        #     emotion_q = self.emotions[song["emotion_pred_p2"]]

                        for i in range(song["num_sequences"]):

                            sequence = {"url": song_rel_url, "index": i, "emotion": emotion_q}
                            self.all_nesmdb_metadata.append(sequence)
                            # sequences_num += 1


    def __getitem__(self, index):
        enc_seq_rel_path = self.all_nesmdb_metadata[index]["url"]
        enc_seq_abs_path = os.path.join(self.encoded_dir, enc_seq_rel_path)

        enc_seq = pickle.load(open(enc_seq_abs_path, "rb"))
        enc_seq = enc_seq[self.all_nesmdb_metadata[index]["index"]]


        if self.transform:
            enc_seq = self.transform(enc_seq, self.dmin, self.dmax)

        emotion = self.all_nesmdb_metadata[index]["emotion"]

        return enc_seq, emotion

    def __len__(self):
        return len(self.all_nesmdb_metadata)
