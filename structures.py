"""Load public pocket–ligand complexes and turn them into graphs."""

import urllib.request
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

AA_DICT = {
    "ALA": 0, "CYS": 1, "ASP": 2, "GLU": 3, "PHE": 4,
    "GLY": 5, "HIS": 6, "ILE": 7, "LYS": 8, "LEU": 9,
    "MET": 10, "ASN": 11, "PRO": 12, "GLN": 13, "ARG": 14,
    "SER": 15, "THR": 16, "VAL": 17, "TRP": 18, "TYR": 19,
    "UNK": 20,
}
AA_PROPERTIES = {
    "ALA": [1.8, 0, 0, 88.6], "CYS": [2.5, 0, 1, 108.5],
    "ASP": [-3.5, -1, 1, 111.1], "GLU": [-3.5, -1, 1, 138.4],
    "PHE": [2.8, 0, 0, 189.9], "GLY": [-0.4, 0, 0, 60.1],
    "HIS": [-3.2, 0.5, 1, 153.2], "ILE": [4.5, 0, 0, 166.7],
    "LYS": [-3.9, 1, 1, 168.6], "LEU": [3.8, 0, 0, 166.7],
    "MET": [1.9, 0, 0, 162.9], "ASN": [-3.5, 0, 1, 114.1],
    "PRO": [-1.6, 0, 0, 112.7], "GLN": [-3.5, 0, 1, 143.8],
    "ARG": [-4.5, 1, 1, 173.4], "SER": [-0.8, 0, 1, 89.0],
    "THR": [-0.7, 0, 1, 116.1], "VAL": [4.2, 0, 0, 140.0],
    "TRP": [-0.9, 0, 0, 227.8], "TYR": [-1.3, 0, 1, 193.6],
    "UNK": [0, 0, 0, 100],
}

# Residue names were read from the PDB HET records. 1HWK is the atorvastatin
# complex: the statin is residue 117, and the ADP in that file is a cofactor.
# 3LZA, 4AKE, 1ALC, and 1TRZ are omitted because they do not contain the
# ligands an earlier version of this repository claimed.
# split is by whole complex. A test protein never appears in training.
COMPLEXES = [
    ("1ATP", "ATP", "cAMP-dependent protein kinase", "train"),
    ("1HCK", "ATP", "cyclin-dependent kinase 2", "train"),
    ("1CSN", "ATP", "casein kinase 1", "train"),
    ("1BYQ", "ADP", "Hsp90", "train"),
    ("1AKE", "AP5", "adenylate kinase", "train"),
    ("1MBO", "HEM", "myoglobin", "train"),
    ("1HDA", "HEM", "hemoglobin", "train"),
    ("2HCK", "QUE", "Src-family kinase Hck", "train"),
    ("1HSG", "MK1", "HIV protease", "train"),
    ("3ERT", "OHT", "estrogen receptor", "train"),
    ("1HWK", "117", "HMG-CoA reductase", "train"),
    ("1EVE", "E20", "acetylcholinesterase", "train"),
    ("1DV2", "ATP", "biotin carboxylase", "test"),
    ("1KAX", "ATP", "Hsp70", "test"),
    ("1BG2", "ADP", "kinesin", "test"),
    ("1ECD", "HEM", "erythrocruorin", "test"),
    ("1IEP", "STI", "Abl kinase", "test"),
    ("1M17", "AQ4", "EGFR kinase", "test"),
    ("1IR3", "ANP", "insulin receptor kinase", "test"),
]

ELEMENTS = ["C", "N", "O", "S", "P", "X"]
SOLVENT = {"HOH", "DOD", "WAT", "H2O"}
RAW_DIR = Path("data/raw")


def download_pdb(pdb_id, save_dir=RAW_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{pdb_id}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())
    return path


def _element(atom):
    element = (atom.element or "").strip().upper()
    if not element:
        element = "".join(c for c in atom.get_name() if c.isalpha())[:1].upper()
    return element if element in ELEMENTS else "X"


def _one_residue(groups):
    """PDB entries often contain several copies. Keep one ligand residue."""
    heavy = []
    for key, atoms in groups.items():
        kept = [atom for atom in atoms if _element(atom) != "H"]
        if len(kept) >= 5:
            heavy.append((key, kept))
    if not heavy:
        return []
    heavy.sort(key=lambda item: len(item[1]), reverse=True)
    return heavy[0][1]


def ligand_atoms(structure, ligand_name):
    groups = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() != ligand_name:
                    continue
                if residue.get_resname().strip() in SOLVENT:
                    continue
                groups.setdefault(residue.get_full_id(), []).extend(residue.get_atoms())
    return _one_residue(groups)


def pocket_residues(structure, ligand, radius=8.0):
    ligand_xyz = np.array([atom.coord for atom in ligand], dtype=np.float32)
    chosen = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" not in residue:
                    continue
                ca = residue["CA"].coord
                if np.linalg.norm(ligand_xyz - ca, axis=1).min() <= radius:
                    chosen.append(residue)
    return chosen


def _edges(coords, cutoff):
    if len(coords) == 1:
        return np.array([[0], [0]], dtype=np.int64)
    pairs = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                pairs.append((i, j))
                pairs.append((j, i))
    if not pairs:
        return np.array([[i, i] for i in range(len(coords))], dtype=np.int64).T
    return np.array(pairs, dtype=np.int64).T


def pocket_graph(residues, cutoff=10.0):
    names, coords = [], []
    for residue in residues:
        names.append(residue.get_resname())
        coords.append(residue["CA"].coord)
    coords = np.array(coords, dtype=np.float32)
    features = []
    for name in names:
        one_hot = np.zeros(21, dtype=np.float32)
        one_hot[AA_DICT.get(name, AA_DICT["UNK"])] = 1.0
        props = np.array(AA_PROPERTIES.get(name, AA_PROPERTIES["UNK"]), dtype=np.float32)
        features.append(np.concatenate([one_hot, props]))
    return {
        "x": np.stack(features),
        "coords": coords,
        "edge_index": _edges(coords, cutoff),
    }


def ligand_graph(atoms, cutoff=2.2):
    coords, features = [], []
    for atom in atoms:
        if _element(atom) == "H":
            continue
        coords.append(atom.coord)
        one_hot = np.zeros(len(ELEMENTS), dtype=np.float32)
        one_hot[ELEMENTS.index(_element(atom))] = 1.0
        features.append(one_hot)
    coords = np.array(coords, dtype=np.float32)
    return {
        "x": np.stack(features),
        "coords": coords - coords.mean(axis=0, keepdims=True),
        "edge_index": _edges(coords, cutoff),
    }


def load_complex(pdb_id, ligand_name, save_dir=RAW_DIR):
    path = download_pdb(pdb_id, save_dir)
    structure = PDBParser(QUIET=True).get_structure(pdb_id, str(path))
    ligand = ligand_atoms(structure, ligand_name)
    if len(ligand) < 5:
        raise ValueError(f"{pdb_id} has no {ligand_name} residue with enough atoms")
    pocket = pocket_residues(structure, ligand)
    if len(pocket) < 8:
        raise ValueError(f"{pdb_id} pocket has only {len(pocket)} residues")
    return {
        "pdb_id": pdb_id,
        "ligand_name": ligand_name,
        "pocket": pocket_graph(pocket),
        "ligand": ligand_graph(ligand),
    }


def load_all(save_dir=RAW_DIR):
    complexes = []
    for pdb_id, ligand_name, title, split in COMPLEXES:
        item = load_complex(pdb_id, ligand_name, save_dir)
        item["title"] = title
        item["split"] = split
        complexes.append(item)
        print(
            f"{split} {pdb_id} {ligand_name}: "
            f"{len(item['pocket']['coords'])} pocket residues, "
            f"{len(item['ligand']['coords'])} ligand atoms"
        )
    return complexes
