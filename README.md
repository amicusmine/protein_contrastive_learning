# Pocket contrastive learning

A small PyTorch example of two tasks on public PDB complexes:

- contrastive retrieval, where a pocket embedding should rank its own ligand above the other ligands
- pocket-conditioned denoising, where ligand coordinates are rebuilt from noise

The set has 124 complexes. Ninety fit the models. Thirty-four are held out by protein: a test PDB never appears in a training step. Ligand residue names were read from the deposited PDB files. Where one protein has several entries, those entries stay in the same split.

```bash
pip install torch biopython numpy matplotlib pandas
python train_demo.py
```

The first run downloads structures from RCSB into `data/raw/`. That directory is gitignored. If a direct download fails, the loader retries through `https_proxy` or `http://127.0.0.1:7897`. Training uses CUDA when it is available, then Apple Metal, then CPU.

Defaults are 800 retrieval steps, learning rate `3e-4`, and seed 0. Denoising steps default to 12 times the number of training complexes.

## How a complex is read

A PDB file often contains several copies of one ligand. The loader keeps a single residue with the requested name and at least five heavy atoms, choosing the copy with the most heavy atoms. The pocket is the standard residues whose Cα lies within 8 Å of any of those ligand atoms.

Pocket nodes are a 21-way amino-acid encoding plus hydrophobicity, charge, polarity, and volume. Ligand nodes are a six-way element encoding (`C`, `N`, `O`, `S`, `P`, other). Ligand coordinates are centered. Edges connect pocket Cα atoms within 10 Å, and ligand atoms within 2.2 Å.

`1HWK` is the atorvastatin complex. The statin is residue `117`. ADP in that file is a cofactor and is not the ligand used here.

## Models

Each encoder is a two-layer mean-aggregation network with a 64-dimensional L2-normalized readout. Retrieval uses symmetric InfoNCE at temperature 0.07. Gradients are clipped at 1.0. Each retrieval step uses the full training set as one batch.

The denoiser is an MLP. Its input is the ligand element, the noisy coordinates, the noise scale in angstroms, and the frozen pocket vector. It predicts the clean coordinates. Training adds `Uniform(0.1, 2.0)` Å of noise. Sampling starts from random coordinates and replaces them at 2.0, 1.0, 0.5, and 0.2 Å. The reported error is a Kabsch RMSD after centering both point clouds.

```
models/retriever.pt
models/denoiser.pt
models/metrics.json
```

## One Apple GPU run

This machine has an M3 Pro and no NVIDIA GPU. The run below used Metal (`mps`), seed 0, 800 retrieval steps, and 1080 denoising steps.

On the 90 training ligands, exact top-1 is 5/90. A random guess among 90 ligands would be right about 1 time in 90.

On the 34 held-out pockets, exact top-1 inside the test gallery is 1/34. A random guess among 34 ligands would be right about 1 time in 34. Against all 124 ligands, the same-ligand top-1 is 2/34: `1C83` ranks its own ligand first, and `1KAX` ranks `1DV2` ATP first. The mean exact rank of a held-out ligand in the gallery of 124 is 50.

Mean Kabsch RMSD is 4.62 Å on the training complexes and 4.39 Å on the held-out complexes.

In the loss figure, the pale line is the loss at each step. The dark line is a moving average: 25 steps for retrieval and 60 steps for coordinate denoising.

![Retrieval and denoising loss](figures/loss.png)

![Held-out pockets against every ligand](figures/retrieval_similarity.png)

![Denoised coordinate error](figures/denoising_rmsd.png)

### Held-out complexes

| PDB | Code | Protein | Rank in 124 | Top hit | RMSD (Å) |
| --- | --- | --- | --- | --- | --- |
| 1DV2 | ATP | biotin carboxylase | 22 | 1C83 OAI | 5.04 |
| 1KAX | ATP | Hsp70 | 9 | 1DV2 ATP | 4.94 |
| 1BG2 | ADP | kinesin | 18 | 1DV2 ATP | 4.07 |
| 1ECD | HEM | erythrocruorin | 10 | 1BMK SB5 | 4.81 |
| 1IR3 | ANP | insulin receptor kinase | 21 | 1HDX NAD | 4.46 |
| 1M17 | AQ4 | EGFR kinase | 47 | 3PTB BEN | 5.26 |
| 1IEP | STI | Abl kinase | 34 | 2FGI PD1 | 6.72 |
| 1CX2 | S58 | COX-2 | 112 | 1C14 TCL | 4.46 |
| 4DFR | MTX | dihydrofolate reductase | 79 | 3ERT OHT | 4.83 |
| 1DWD | MID | thrombin | 55 | 1DV2 ATP | 4.44 |
| 3HS4 | AZM | carbonic anhydrase II | 92 | 3PTB BEN | 3.10 |
| 2QWK | G39 | neuraminidase | 30 | 1QHA ANP | 3.19 |
| 2PRG | BRL | PPAR gamma | 9 | 3PTB BEN | 4.80 |
| 1O86 | LPR | ACE | 34 | 5TMN 0PJ | 4.64 |
| 1X70 | 715 | DPP-4 | 28 | 1C83 OAI | 5.04 |
| 1UK0 | FRM | PARP | 32 | 4TMN 0PK | 5.06 |
| 1W51 | L01 | BACE | 56 | 1DV2 ATP | 5.11 |
| 1C83 | OAI | PTP1B | 1 | 1C83 OAI | 3.47 |
| 121P | GCP | H-Ras | 6 | 1HDX NAD | 4.81 |
| 1F88 | RET | rhodopsin | 38 | 1W0E MET | 4.30 |
| 1JFF | TA1 | tubulin | 107 | 2B7A IZA | 5.15 |
| 3EQM | ASD | aromatase | 14 | 3EML ZMA | 3.83 |
| 2V5Z | SAG | monoamine oxidase B | 53 | 1AH3 TOL | 4.76 |
| 1KSN | FXV | factor Xa | 90 | 1ICE ASA | 5.27 |
| 1GFW | MSI | caspase-3 | 70 | 1C83 OAI | 4.09 |
| 1T64 | TSN | HDAC | 113 | 1C83 OAI | 4.33 |
| 2AM9 | TES | androgen receptor | 15 | 1AH3 TOL | 3.46 |
| 1W0E | MET | CYP3A4 | 87 | 3PTB BEN | 2.18 |
| 2RH1 | CAU | beta2 adrenergic receptor | 34 | 1W0E MET | 3.87 |
| 3PTB | BEN | trypsin | 115 | 1C83 OAI | 1.68 |
| 1HRC | HEC | cytochrome c | 10 | 1A9U SB2 | 5.15 |
| 2H42 | VIA | PDE5 | 85 | 1AH3 TOL | 4.35 |
| 1QS4 | 100 | HIV integrase | 110 | 1IR3 ANP | 3.89 |
| 2OC8 | U5G | HCV protease | 61 | 1W0E MET | 4.64 |

The same rows are in `data/processed/heldout_results.csv`.

### Training complexes

| PDB | Code | Protein |
| --- | --- | --- |
| 1ATP | ATP | cAMP-dependent protein kinase |
| 1STC | STU | cAMP-dependent protein kinase |
| 1FMO | ADN | cAMP-dependent protein kinase |
| 1HCK | ATP | cyclin-dependent kinase 2 |
| 1FIN | ATP | cyclin-dependent kinase 2 |
| 1AQ1 | STU | cyclin-dependent kinase 2 |
| 1CKP | PVB | cyclin-dependent kinase 2 |
| 2C6O | 4SP | cyclin-dependent kinase 2 |
| 1CSN | ATP | casein kinase 1 |
| 1PHK | ATP | phosphorylase kinase |
| 1JWH | ANP | casein kinase 2 |
| 1A9U | SB2 | p38 MAP kinase |
| 1BMK | SB5 | p38 MAP kinase |
| 1UWH | BAX | B-Raf |
| 1Q3D | STU | GSK3 |
| 1Q41 | IXM | GSK3 |
| 1O6K | ANP | AKT |
| 1XWS | BI1 | Pim-1 |
| 1QPE | PP2 | Lck |
| 2B7A | IZA | JAK2 |
| 1T46 | STI | Kit |
| 2FGI | PD1 | FGFR1 |
| 1AGW | SU2 | FGFR1 |
| 1NVQ | UCN | Chk1 |
| 1XBB | STI | Syk |
| 1U59 | STU | ZAP-70 |
| 2SRC | ANP | Src |
| 2RFS | AM8 | c-Met |
| 1Y6A | AAZ | VEGFR2 |
| 1PY5 | PY1 | TGF-beta receptor |
| 1SM2 | STU | Itk |
| 1BYQ | ADP | Hsp90 |
| 1UY6 | PU3 | Hsp90 |
| 1AKE | AP5 | adenylate kinase |
| 1ZIN | AP5 | adenylate kinase |
| 1MBO | HEM | myoglobin |
| 1MBN | HEM | myoglobin |
| 1HDA | HEM | hemoglobin |
| 1HHO | HEM | hemoglobin |
| 2HHB | HEM | hemoglobin |
| 2HCK | QUE | Src-family kinase Hck |
| 1HSG | MK1 | HIV protease |
| 1HVR | XK2 | HIV protease |
| 1HPX | KNI | HIV protease |
| 4PHV | VAC | HIV protease |
| 3ERT | OHT | estrogen receptor |
| 1ERE | EST | estrogen receptor |
| 1ERR | RAL | estrogen receptor |
| 1HWK | 117 | HMG-CoA reductase |
| 1EVE | E20 | acetylcholinesterase |
| 1EQG | IBP | COX-1 |
| 1PGG | IMM | COX-1 |
| 1OG5 | SWF | CYP2C9 |
| 1R9O | FLP | CYP2C9 |
| 1HDX | NAD | alcohol dehydrogenase |
| 1I10 | NAI | lactate dehydrogenase |
| 1AH3 | TOL | aldose reductase |
| 1EL3 | I84 | aldose reductase |
| 1HFC | PLH | matrix metalloproteinase |
| 1CGL | 0ED | collagenase |
| 4TMN | 0PK | thermolysin |
| 5TMN | 0PJ | thermolysin |
| 1ICE | ASA | caspase-1 |
| 1CSB | EP0 | cathepsin B |
| 1MEM | 0D6 | cathepsin K |
| 2V0Z | C41 | renin |
| 1VRT | NVP | HIV reverse transcriptase |
| 1FK9 | EFZ | HIV reverse transcriptase |
| 1C14 | TCL | enoyl-ACP reductase |
| 1P44 | GEQ | InhA |
| 1KZN | CBN | DNA gyrase |
| 1CBS | REA | CRABP |
| 1DB1 | VDX | vitamin D receptor |
| 1FBY | 9CR | RXR |
| 1A28 | STR | progesterone receptor |
| 1M2Z | DEX | glucocorticoid receptor |
| 1K7L | 544 | PPAR alpha |
| 1I7I | AZ2 | PPAR delta |
| 1P8D | CO1 | LXR |
| 1OSV | CHC | FXR |
| 1FTN | GDP | RhoA |
| 1GP2 | GDP | G protein alpha |
| 2YDO | ADN | adenosine A2A receptor |
| 3EML | ZMA | adenosine A2A receptor |
| 1OYN | ROL | PDE4 |
| 1VID | DNC | COMT |
| 1MMD | ADP | myosin |
| 4PFK | ADP | phosphofructokinase |
| 1A49 | ATP | pyruvate kinase |
| 1QHA | ANP | aspartate carbamoyltransferase |

## Binding-site pairs

`data_preparation.py`, `train.py`, `evaluate.py`, and `demo.py` are an older site-versus-site example. Positive pairs share a ligand residue name. Pairs are built inside one split, so a test protein does not appear in `train.csv` or `val.csv`. `val.csv` is a pair split of the training proteins. The held-out numbers above come from `train_demo.py`.

`train.py` needs PyTorch Geometric as well as PyTorch. If `pip install -r requirements.txt` fails on `torch-geometric`, `torch-scatter`, or `torch-sparse`, use the wheel command from the [PyG install page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
python data_preparation.py
python train.py --epochs 50 --batch_size 8
python evaluate.py --checkpoint checkpoints/best_model.pt
```

## Scope

The code in this repository trains only on the public structures listed above. The training top-1 shows a weak fit to those 90 pairs. The held-out top-1 stays close to a random guess among the test ligands. The RMSD near 4.4 Å is the coordinate error of this denoiser.
