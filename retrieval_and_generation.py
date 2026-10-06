"""Toy pocket–ligand retrieval and pocket-conditioned coordinate denoising.

This is an independent sketch of two tasks, trained on eight public complexes.
It is not a reimplementation of a full screening or diffusion model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_mean(src, index, size):
    total = torch.zeros(size, src.size(-1), device=src.device, dtype=src.dtype)
    total.index_add_(0, index, src)
    counts = torch.zeros(size, device=src.device, dtype=src.dtype)
    counts.index_add_(0, index, torch.ones(index.shape[0], device=src.device, dtype=src.dtype))
    return total / counts.clamp(min=1).unsqueeze(-1)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64, layers=2):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.messages = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.updates = nn.ModuleList(nn.Linear(hidden * 2, hidden) for _ in range(layers))
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.input(x))
        src, dst = edge_index
        for message, update in zip(self.messages, self.updates):
            pooled = _scatter_mean(message(h)[src], dst, h.size(0))
            h = F.relu(update(torch.cat([h, pooled], dim=-1)))
        return F.normalize(self.output(h.mean(dim=0)), dim=-1)


class PocketLigandRetriever(nn.Module):
    def __init__(self, dim=64, temperature=0.07):
        super().__init__()
        self.pocket = GraphEncoder(25, out_dim=dim)
        self.ligand = GraphEncoder(6, out_dim=dim)
        self.temperature = temperature

    def encode(self, pocket, ligand):
        return self.pocket(pocket["x"], pocket["edge_index"]), self.ligand(
            ligand["x"], ligand["edge_index"]
        )

    def loss(self, pocket_emb, ligand_emb):
        logits = pocket_emb @ ligand_emb.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )


class CoordinateDenoiser(nn.Module):
    """Predict the noise added to ligand coordinates, conditioned on a pocket vector."""

    def __init__(self, pocket_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6 + 3 + 1 + pocket_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, element, coords, time, pocket):
        t = time.expand(element.size(0), 1)
        cond = pocket.expand(element.size(0), -1)
        return self.net(torch.cat([element, coords, t, cond], dim=-1))


def tensors(graph, device):
    return {
        "x": torch.tensor(graph["x"], dtype=torch.float32, device=device),
        "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long, device=device),
        "coords": torch.tensor(graph["coords"], dtype=torch.float32, device=device),
    }


def retrieval_batch(model, complexes, device):
    pockets, ligands = [], []
    for item in complexes:
        pocket, ligand = model.encode(
            tensors(item["pocket"], device), tensors(item["ligand"], device)
        )
        pockets.append(pocket)
        ligands.append(ligand)
    return torch.stack(pockets), torch.stack(ligands)


def ranking_table(pocket_emb, ligand_emb, complexes):
    similarity = pocket_emb @ ligand_emb.T
    order = similarity.argsort(dim=1, descending=True)
    lines = []
    top1 = 0
    for i, item in enumerate(complexes):
        rank = int((order[i] == i).nonzero()[0]) + 1
        top1 += rank == 1
        guess = complexes[int(order[i, 0])]
        lines.append(
            f"{item['pdb_id']} {item['ligand_name']}: "
            f"true rank {rank}/{len(complexes)}, "
            f"top hit {guess['pdb_id']} {guess['ligand_name']}"
        )
    return top1 / len(complexes), lines


def denoise_loss(denoiser, pocket_emb, ligand, device):
    """Predict the clean coordinates from a noised copy. Noise scale is in angstroms."""
    element = tensors(ligand, device)["x"]
    clean = tensors(ligand, device)["coords"]
    sigma = torch.empty((), device=device).uniform_(0.1, 2.0)
    noisy = clean + torch.randn_like(clean) * sigma
    predicted = denoiser(element, noisy, sigma.view(1), pocket_emb.detach())
    return F.mse_loss(predicted, clean)


@torch.no_grad()
def sample_coords(denoiser, pocket_emb, ligand, device):
    element = tensors(ligand, device)["x"]
    coords = torch.randn(element.size(0), 3, device=device)
    for sigma in (2.0, 1.0, 0.5, 0.2):
        time = torch.tensor([sigma], device=device)
        coords = denoiser(element, coords, time, pocket_emb)
    return coords.cpu().numpy()


def kabsch_rmsd(pred, true):
    pred = pred - pred.mean(axis=0)
    true = true - true.mean(axis=0)
    covariance = pred.T @ true
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    aligned = pred @ rotation
    return float(np.sqrt(((aligned - true) ** 2).sum(axis=1).mean()))
