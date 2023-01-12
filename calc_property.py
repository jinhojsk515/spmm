from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors
import torch
from rdkit import RDLogger


from inspect import getmembers, isfunction
from rdkit.Chem import Descriptors

descriptor_dict = {}
for i in getmembers(Descriptors, isfunction):
    if 'AUTOCORR2D' in i[0]:    pass
    elif 'Ipc' in i[0]:  pass
    elif 'VSA' in i[0]: pass
    elif i[0] == '_ChargeDescriptors':  break
    else:
        descriptor_dict[i[0]] = i[1]
        #print(i[0])    #name of properties



def calculate_property(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    output = []
    for i,descriptor in enumerate(descriptor_dict):
        if i not in [0, 1, 2, 3, 4, 5, 6, 7, 35, 37, 39, 41]:
            #print(descriptor)
            output.append(descriptor_dict[descriptor](mol))
    output.append(Chem.QED.qed(mol))
    return torch.tensor(output,dtype=torch.float)


def calculate_property_small(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    output=[]
    output.append(rdMolDescriptors.CalcNumHBD(mol))
    output.append(rdMolDescriptors.CalcNumHBA(mol))
    output.append(rdMolDescriptors.CalcExactMolWt(mol))
    output.append(rdMolDescriptors.CalcNumRings(mol))
    output.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    output.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    output.append(rdMolDescriptors.CalcTPSA(mol))
    output.append(mol.GetNumAtoms())
    output.append(mol.GetNumBonds())
    output.append(Descriptors.MolLogP(mol))
    output.append(Chem.Crippen.MolMR(mol))  #already in Descriptors
    output.append(Chem.QED.qed(mol))
    return torch.tensor(output,dtype=torch.float)

if __name__=='__main__':
    print(calculate_property('Cc1cccc(CNNC(=O)C2(Cc3ccccc3CN=[N+]=[N-])N=C(c3ccc(OCCCO)cc3)OC2c2ccc(Br)cc2)c1'))
