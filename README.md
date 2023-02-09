# SPMM: Structure-Property Multi-Modal foundation model

VLP model for chemical domain, using molecule structure and chemical properties.

Molecule structure will be given in SMILES, and we used 53 simple cheimical properties to build a property vector(PV) of a molecule.

!! Only Pretrain, Molecule generation and Property generation are here, which can be run with checkpoint `.pt` file only and without any data. Downstream tasks which requires additional training(ex: molecule property prediction) are not included. 


## Code running
Arguments can be passed with commands, or edited manually in the running code. But default values are already good to go, unless you want to modify the model size.


-Pretrain

```
python pretrain.py
```

-Property generation: The model takes the query molecules in `input_file`, and generate their 53 chemical properties that are used in the pre-training process.

```
python pg.py --checkpoint './Pretrain/checkpointname.pth' --input_file './data/datafilename.txt'
```

-Molecule generation: The model takes one query PV(contains 53 properties), and generate `n_generate` molecules that satisfies the input PV condition. The generated molecules will be written in `generated_molecules.txt`.

```
python mmg.py --checkpoint './Pretrain/checkpointname.pth' --n_generate 100
```

Since passing all 53 chemical properties by argument is too clumsy, at this moment, you have to manually build the input PV in the code. Maybe you could modify the code to read .txt file to build input PV.
