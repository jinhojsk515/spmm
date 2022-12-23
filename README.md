# SPMM
VLP for chemical domain, using molecule structure and chemical properties.

Molecule structure will be given in SMILES, and we used 53 simple cheimical properties to build a property vector(PV) of a molecule.

!! Only Molecule generation and Property generation are here, which can be run with checkpoint'.pt' file only and without any data. Downstream tasks which requires additional training(ex: molecule property prediction) are not included. 

## File explanation

## Code running
Arguments can be passed with commands, or edited manually in the running code. But default values are already good to go, unless you want to modify the model size.


-Property generation: The model takes the query molecules in , and generate its 53 chemical properties that are used in the pre-training process.

```
python pg.py --checkpoint './Pretrain/checkpoint_08.pth' --input_file './data/pubchem-10m-simple.txt'
```

-Molecule generation: The model takes one query PV(contains 53 properties), and generate 'n_generate' molecules that satisfies the input PV condition. The generated molecules are written in 'generated_molecules.txt'.

```
python mmg.py --checkpoint './Pretrain/checkpoint_08.pth' --n_generate 100
```

Since passing all 53 chemical properties by argument is too clumsy, at this moment, you have to manually build the input PV in the code. Maybe you could modify the code to read .txt file to build input PV.
