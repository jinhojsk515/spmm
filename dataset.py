from torch.utils.data import Dataset
import torch
from calc_property import calculate_property
import random
import pandas as pd
from rdkit import Chem
import pickle


class SMILESDataset_pretrain(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        with open(data_path,'r') as f:  lines=f.readlines()
        self.data=[l.strip() for l in lines]

        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # smiles ='Q'+self.data[index]
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]), isomericSmiles=False, canonical=True)
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std

        return properties, smiles
