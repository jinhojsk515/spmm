# SPMM: Structure-Property Multi-Modal learning for molecules

The official GitHub for SPMM, a multi-modal molecular pre-train model for a synergistic comprehension of molecular structure and properties.
The details can be found in the following paper: 
Bidirectional Generation of Structure and Properties Through a Single Molecular Foundation Model.
https://arxiv.org/abs/2211.10590

[![DOI](https://zenodo.org/badge/542878783.svg)](https://zenodo.org/doi/10.5281/zenodo.10567598)

***<ins>The model checkpoint and data are too heavy to be included in this repo, and they can be found [here](https://drive.google.com/drive/folders/1ARrSg9kXdXAL5VGgDBwizpSgcJwauPua?usp=sharing).<ins>***

![method1](https://github.com/jinhojsk515/SPMM/assets/59189526/1ff52950-aa12-481f-94ea-4d1e97ac7bf3)

Molecule structure will be given in SMILES, and we used 53 simple chemical properties to build a property vector(PV) of a molecule.

## File description
* `data/`: Contains the data used for the experiments in the paper. (you have to make this folder and put the data that you downloaded from the link above.)
* `Pretrain/`: Contains the checkpoint of the pre-trained SPMM. (you have to make this folder and put the checkpoint that you downloaded from the link above.)
* `vocab_bpe_300.txt`: Contains the SMILES tokens for the SMILES tokenizer.
* `property_name.txt`: Contains the name of the 53 chemical properties.
* `normalize.pkl`: Contains the mean and standard deviation of the 53 chemical properties that we used for PV.
* `calc_property.py`: Contains the code to calculate the 53 chemical properties and build a PV for a given SMILES. **Modify this code accordingly to utilize SPMM pre-training for your custom PVs.**
* `SPMM_models.py`: Contains the code for the SPMM model and its pre-training codes.
* `SPMM_pretrain.py`: runs SPMM pre-training.
* `d_*.py`: Codes for the downstream tasks.

## Requirements
Run `pip install -r requirements.txt` to install the required packages.

## Code running
Arguments can be passed with commands, or be edited manually in the running code.

1. Pre-training
    ```
    python SPMM_pretrain.py --data_path './data/pretrain_50m.txt'
    ```

2. PV-to-SMILES generation
   * deterministic: The model takes PVs from the molecules in `input_file`, and generate molecules with those PVs using k-beam search. The generated molecules will be written in `generated_molecules.txt`.
       ```
       python d_pv2smiles_deterministic.py --checkpoint './Pretrain/checkpoint_SPMM_50m.ckpt' --input_file './data/pubchem_1k_unseen.txt' --k 2
       ```
   * stochastic: The model takes one query PV and generate `n_generate` molecules with that PV. The generated molecules will be written in `generated_molecules.txt`. Here, you need to build your input PV in the code. Check four examples that we included.
       ```
       python d_pv2smiles_stochastic.py --checkpoint './Pretrain/checkpoint_SPMM_50m.ckpt' --n_generate 1000 --stochastic True --k 2
       ```

3. SMILES-to-PV generation
    
    The model takes the query molecules in `input_file`, and generate their PV.

    ```
    python d_smiles2pv.py --checkpoint './Pretrain/checkpoint_SPMM_50m.ckpt' --input_file './data/pubchem_1k_unseen.txt'
    ```

4. MoleculeNet + DILI prediction task

    `d_regression.py`, `d_classification.py`, and `d_classification_multilabel.py`, performs regression, binary classification, and multi-label classification tasks, respectively.

    ```
    python d_regression.py --checkpoint './Pretrain/checkpoint_SPMM_50m.ckpt' --name 'esol'
    python d_classification.py --checkpoint './Pretrain/checkpoint_SPMM_50m.ckpt' --name 'bbbp'
    python d_classification_multilabel.py --checkpoint './Pretrain/checkpoint_SPMM_50m.ckpt' --name 'clintox'
    ```

5. Forward/retro-reaction prediction tasks

    `d_rxn_prediction.py` performs both forward/reverse reaction prediction task on USPTO-480k and USPTO-50k dataset.

    e.g. forward reaction prediction, no beam search
    ```
    python d_rxn_prediction.py --checkpoint './Pretrain/checkpoint_SPMM_50m.ckpt' --mode 'forward' --n_beam 1 
    ```
    e.g. retro reaction prediction, beam search with k=3
    ```
    python d_rxn_prediction.py --checkpoint './Pretrain/checkpoint_SPMM_50m.ckpt' --mode 'retro' --n_beam 3 
    ```

## Acknowledgement
* The code for BERT with cross-attention layers `xbert.py` and schedulers are modified from the one in [ALBEF](https://github.com/salesforce/ALBEF).
* The code for SMILES augmentation is taken from [pysmilesutils](https://github.com/MolecularAI/pysmilesutils).
