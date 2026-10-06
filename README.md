# Pocket contrastive learning

A small, runnable example of contrastive learning on protein binding sites.

Sites that bind the same ligand type are pulled together in embedding space. Sites that bind different ligand types are pushed apart. The set is a handful of public PDB entries, not a benchmark. It is meant to show the data flow and the training loop.

## What it does

1. Download PDB files from RCSB.
2. Keep residues within 10 Å of the named ligand.
3. Turn each site into a residue graph.
4. Encode the graph with a graph attention network.
5. Train with InfoNCE.
6. Embed a new site and rank the others by cosine similarity.

Positive pairs come from the same ligand group (ATP, ADP, or heme). Negative pairs come from different groups.

## Layout

```
data_preparation.py    download PDBs, cut sites, write pair splits
model.py               residue graph, GAT encoder, InfoNCE
train.py               training loop
evaluate.py            retrieval metrics and plots from a checkpoint
demo.py                embed one site and retrieve neighbors
```

Downloaded structures go to `data/raw/`. Pair tables go to `data/processed/`. Checkpoints go to `checkpoints/`.

## Setup

Python 3.10 or newer.

```bash
pip install torch
pip install -r requirements.txt
```

`torch-geometric`, `torch-scatter`, and `torch-sparse` must match the installed PyTorch build. If that install fails, use the wheel command from the [PyG install page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) instead of forcing `requirements.txt`.

## Run

From this directory:

```bash
python data_preparation.py
python train.py --epochs 50 --batch_size 8
python evaluate.py --checkpoint checkpoints/best_model.pt
```

`data_preparation.py` downloads this set and labels it by the names below. Some of those names are not the ligand actually deposited in the PDB file, so use `train_demo.py` when the ligand has to be real.

| Label in the script | PDB |
| --- | --- |
| ATP | 1ATP, 3LZA, 2HCK |
| ADP | 1AKE, 4AKE |
| Heme | 1MBO, 1HDA |
| Other | 1ALC, 1TRZ |

Training defaults are in `train.py`: hidden size 128, embedding size 256, learning rate `1e-3`, weight decay `1e-4`, 50 epochs. The checkpoint is `checkpoints/best_model.pt`.

After training, `demo.py` loads that checkpoint and retrieves similar sites.

## Model

Each residue is a node. Node features are a 21-way amino-acid encoding plus four physicochemical values (hydrophobicity, charge, polarity, volume). An edge connects two residues whose Cα atoms are within 10 Å, with the distance and direction as edge features.

The encoder is a 3-layer GAT with 4 heads, batch norm, and dropout. Mean and max pooling are concatenated and projected to a 256-dimensional L2-normalized vector. The loss is InfoNCE with temperature 0.07.

## Pocket–ligand retrieval and coordinate denoising

`train_demo.py` is a separate, smaller example of two tasks on eight public complexes:

- contrastive retrieval, where a pocket and its own ligand should rank above the other ligands
- pocket-conditioned denoising, where ligand atom coordinates are rebuilt from noise using the pocket vector

The ligand names in this list were checked against the PDB files. The older site-only demo above uses a different, rougher label set.

```bash
pip install torch biopython numpy
python train_demo.py
```

This does not use the group codebase, a large screening library, or a full 3D diffusion model. With only eight complexes, a high training-set rank means the loop memorized those pairs.

## Scope

The site-only example uses nine structures. It does not train on PDBbind or sc-PDB, and the splits are random pair splits of that toy set. Numbers from either script are a check that the loop runs.
