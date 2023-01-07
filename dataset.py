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
        #self.property_mean=torch.tensor([  1.4267,   4.2140, 363.0794,   2.7840,   1.9534,   5.6722,  71.2552,
        # 25.0858,  26.8583,   2.7226,  96.8194,   0.6098])
        #self.property_std=torch.tensor([  1.7206,   2.7012, 164.6371,   1.6557,   1.4255,   5.3647,  54.3515,
        # 11.7913,  12.8683,   2.7610,  44.8578,   0.2197])
        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles ='Q'+self.data[index]
        #smiles='Q'+augmentation(self.data[index])
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        #properties=torch.nan_to_num(properties, nan=-100.,posinf=-100.,neginf=-100.)
        #print(properties)

        return properties, smiles


class SMILESDataset_BACE(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False,mode=None,cls=False):
        self.cls=cls
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        self.data = [data.iloc[i] for i in range(len(data)) if data.iloc[i]['Model']==mode]
        print(len(self.data))

        #self.property_mean=torch.tensor([  1.4267,   4.2140, 363.0794,   2.7840,   1.9534,   5.6722,  71.2552,
        # 25.0858,  26.8583,   2.7226,  96.8194,   0.6098])
        #self.property_std=torch.tensor([  1.7206,   2.7012, 164.6371,   1.6557,   1.4255,   5.3647,  54.3515,
        # 11.7913,  12.8683,   2.7610,  44.8578,   0.2197])
        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        self.value_mean=torch.tensor(6.5220)
        self.value_std=torch.tensor(1.3424)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]

class SMILESDataset_BACEC(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        with open('./normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['mol']), isomericSmiles=False,canonical=True)
        value=int(self.data[index]['Class'])
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        #properties += torch.randn_like(properties) * 0.01

        return properties, smiles, value

class SMILESDataset_BACER(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        self.value_mean=torch.tensor(6.420878294545455)
        self.value_std=torch.tensor(1.345219669175284)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        #value=(self.data[index]['exp']-self.value_mean)/self.value_std
        value = torch.tensor(self.data[index]['target'].item())
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        return properties, smiles, value


class SMILESDataset_LIPO(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        #self.property_mean=torch.tensor([  1.4267,   4.2140, 363.0794,   2.7840,   1.9534,   5.6722,  71.2552,
        # 25.0858,  26.8583,   2.7226,  96.8194,   0.6098])
        #self.property_std=torch.tensor([  1.7206,   2.7012, 164.6371,   1.6557,   1.4255,   5.3647,  54.3515,
        # 11.7913,  12.8683,   2.7610,  44.8578,   0.2197])
        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        self.value_mean=torch.tensor(2.162904761904762)
        self.value_std=torch.tensor(1.210992810122257)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        #value=(self.data[index]['exp']-self.value_mean)/self.value_std
        value = torch.tensor(self.data[index]['exp'].item())
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        #properties += torch.randn_like(properties) * 0.01

        return properties, smiles, value


class SMILESDataset_Clearance(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        #self.property_mean=torch.tensor([  1.4267,   4.2140, 363.0794,   2.7840,   1.9534,   5.6722,  71.2552,
        # 25.0858,  26.8583,   2.7226,  96.8194,   0.6098])
        #self.property_std=torch.tensor([  1.7206,   2.7012, 164.6371,   1.6557,   1.4255,   5.3647,  54.3515,
        # 11.7913,  12.8683,   2.7610,  44.8578,   0.2197])
        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        self.value_mean=torch.tensor(51.503692077727955)
        self.value_std=torch.tensor(53.50834365711207)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        #value=(self.data[index]['target']-self.value_mean)/self.value_std
        value = torch.tensor(self.data[index]['target'].item())
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std

        return properties, smiles, value


class SMILESDataset_BBBP(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data)) if Chem.MolFromSmiles(data.iloc[i]['smiles'])!=None] #'.' not in data.iloc[i]['smiles'] and

        #self.property_mean=torch.tensor([  1.4267,   4.2140, 363.0794,   2.7840,   1.9534,   5.6722,  71.2552,
        # 25.0858,  26.8583,   2.7226,  96.8194,   0.6098])
        #self.property_std=torch.tensor([  1.7206,   2.7012, 164.6371,   1.6557,   1.4255,   5.3647,  54.3515,
        # 11.7913,  12.8683,   2.7610,  44.8578,   0.2197])
        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            smiles = 'Q' + Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles'].replace('.',' [SEP] Q')), isomericSmiles=False)
        except:
            print(Chem.MolFromSmiles(self.data[index]['smiles']))
            raise NotImplementedError
        label=int(self.data[index]['p_np'])
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        #properties += torch.randn_like(properties) * 0.01

        return properties, smiles, label


class SMILESDataset_ESOL(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        #self.value_mean=torch.tensor(-2.988192375886525)
        #self.value_std=torch.tensor(1.682473550858686)
        self.value_mean=torch.tensor(-2.8668758314855878)
        self.value_std=torch.tensor(2.066724108076815)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        #value=(self.data[index]['ESOL predicted log solubility in mols per litre']-self.value_mean)/self.value_std
        value = torch.tensor(self.data[index]['ESOL predicted log solubility in mols per litre'].item())
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        #properties += torch.randn_like(properties) * 0.01

        return properties, smiles, value


class SMILESDataset_Freesolv(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        #self.value_mean=torch.tensor(-3.8030062305295944)
        #self.value_std=torch.tensor(3.844822204602953)
        self.value_mean=torch.tensor(-3.2594736842105267)
        self.value_std=torch.tensor(3.2775297233608893)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value=(self.data[index]['target']-self.value_mean)/self.value_std
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        #properties += torch.randn_like(properties) * 0.01

        return properties, smiles, value


class SMILESDataset_HIV(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        data = pd.read_csv(data_path)
        #self.data = [data.iloc[i] for i in range(len(data)) if '.' not in data.iloc[i]['smiles'] and Chem.MolFromSmiles(data.iloc[i]['smiles'])!=None]
        self.data = [data.iloc[i] for i in range(len(data)) if Chem.MolFromSmiles(data.iloc[i]['smiles']) != None]

        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = 'Q' + Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False, canonical=True)
        smiles = smiles.replace('.', ' [SEP] Q')
        label=int(self.data[index]['HIV_active'])
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        return properties, smiles, label



class SMILESDataset_Clintox(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        self.n_output=2

        with open('./normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value=torch.tensor([float(self.data[index]['FDA_APPROVED']),float(self.data[index]['CT_TOX'])])
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        #properties += torch.randn_like(properties) * 0.01

        return properties, smiles, value

class SMILESDataset_QM7(Dataset):
    def __init__(self, data_path, data_length=None,shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        with open('./normalize.pkl', 'rb') as w:  norm = pickle.load(w)
        self.property_mean,self.property_std = norm

        self.value_mean=torch.tensor(-1530.0040274223034)
        self.value_std=torch.tensor(223.82825195108055)
        #self.value_mean = torch.tensor(0)
        #self.value_std = torch.tensor(1)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data=self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles='Q'+Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        #value=(self.data[index]['expt']-self.value_mean)/self.value_std
        value = torch.tensor(self.data[index]['expt'].item())
        properties=(calculate_property(smiles[1:])-self.property_mean)/self.property_std
        #properties += torch.randn_like(properties) * 0.01

        return properties, smiles, value