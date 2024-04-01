from rdkit import Chem
import torch
import rdkit
from rdkit import RDLogger
from inspect import getmembers, isfunction
from rdkit.Chem import Descriptors
import time
from collections import OrderedDict


with open('./property_name.txt', 'r') as f:
    names = [n.strip() for n in f.readlines()][:53]

descriptor_dict = OrderedDict()
for n in names:
    if n == 'QED':
        descriptor_dict[n] = lambda x: Chem.QED.qed(x)
    else:
        descriptor_dict[n] = getattr(Descriptors, n)

def calculate_property(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        # print(descriptor)
        output.append(descriptor_dict[descriptor](mol))
    return torch.tensor(output, dtype=torch.float)


if __name__ == '__main__':
    st = time.time()
    output = calculate_property('Cc1cccc(CNNC(=O)C2(Cc3ccccc3CN=[N+]=[N-])N=C(c3ccc(OCCCO)cc3)OC2c2ccc(Br)cc2)c1')
    print(output, output.size())
    print(time.time() - st)
    print(rdkit.__version__)
