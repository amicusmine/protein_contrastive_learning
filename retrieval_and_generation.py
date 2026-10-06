"""Pocket–ligand retrieval and pocket-conditioned coordinate denoising.

The encoders read distances as well as atom labels. Retrieval positives are
ligands that share a residue name, not only the same PDB entry.
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


def _rbf(dist, d_max, n_rbf):
    centers = np.linspace(0, d_max, n_rbf, dtype=np.float32)
    width = np.float32(d_max / max(n_rbf - 1, 1))
    return np.exp(-0.5 * ((dist[:, None] - centers) / width) ** 2).astype(np.float32)


def _shape(coords, hist_bins):
    coords = np.asarray(coords, dtype=np.float32)
    if len(coords) < 2:
        return np.zeros(hist_bins, dtype=np.float32)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    values = dist[~np.eye(len(coords), dtype=bool)]
    centers = np.linspace(0, 16.0, hist_bins, dtype=np.float32)
    width = np.float32(16.0 / max(hist_bins - 1, 1))
    weights = np.exp(-0.5 * ((values[:, None] - centers) / width) ** 2)
    hist = weights.sum(axis=0)
    total = float(hist.sum())
    if total <= 0:
        return np.zeros(hist_bins, dtype=np.float32)
    return (hist / total).astype(np.float32)


def _attach_graph(graph, d_max, n_rbf, hist_bins):
    if "edge_rbf" in graph:
        return
    coords = np.asarray(graph["coords"], dtype=np.float32)
    src, dst = graph["edge_index"]
    dist = np.linalg.norm(coords[src] - coords[dst], axis=1).astype(np.float32)
    graph["degree"] = np.bincount(dst, minlength=len(coords)).astype(np.float32)
    graph["edge_rbf"] = _rbf(dist, d_max, n_rbf)
    graph["shape"] = _shape(coords, hist_bins)


def attach_geometry(complexes, pocket_d_max=10.0, ligand_d_max=2.5, n_rbf=8, hist_bins=16):
    for item in complexes:
        _attach_graph(item["pocket"], pocket_d_max, n_rbf, hist_bins)
        _attach_graph(item["ligand"], ligand_d_max, n_rbf, hist_bins)
    return complexes


class GraphEncoder(nn.Module):
    """Mean-aggregation encoder with distance features and a shape readout."""

    def __init__(self, in_dim, hidden=64, out_dim=64, layers=2, n_rbf=8, d_max=10.0, hist_bins=16):
        super().__init__()
        self.n_rbf = n_rbf
        self.d_max = float(d_max)
        self.hist_bins = hist_bins
        self.input = nn.Linear(in_dim + 1, hidden)
        self.messages = nn.ModuleList(nn.Linear(hidden + n_rbf, hidden) for _ in range(layers))
        self.updates = nn.ModuleList(nn.Linear(hidden * 2, hidden) for _ in range(layers))
        self.output = nn.Linear(hidden * 2 + 1 + hist_bins, out_dim)

    def forward(self, x, edge_index, degree, edge_rbf, shape, batch, n_graphs):
        src, dst = edge_index
        h = F.relu(self.input(torch.cat([x, (degree / 10.0).unsqueeze(-1)], dim=-1)))
        for message, update in zip(self.messages, self.updates):
            sent = message(torch.cat([h[src], edge_rbf], dim=-1))
            pooled = _scatter_mean(sent, dst, h.size(0))
            h = F.relu(update(torch.cat([h, pooled], dim=-1)))
        mean = _scatter_mean(h, batch, n_graphs)
        maxed = h.new_full((n_graphs, h.size(-1)), -1e9)
        maxed.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce="amax", include_self=True)
        counts = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch, torch.ones(h.size(0), device=h.device, dtype=h.dtype))
        size = torch.log(counts.clamp(min=1)).unsqueeze(-1)
        graph = torch.cat([mean, maxed, size, shape], dim=-1)
        return F.normalize(self.output(graph), dim=-1)


def same_name_mask(names, device):
    groups = {}
    for index, name in enumerate(names):
        groups.setdefault(name, []).append(index)
    n = len(names)
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    for indexes in groups.values():
        idx = torch.tensor(indexes, device=device)
        mask[idx[:, None], idx[None, :]] = True
    return mask


def _nce_rows(logits, mask):
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    kept = log_prob.masked_fill(~mask, 0)
    counts = mask.sum(dim=1).clamp(min=1)
    return -(kept.sum(dim=1) / counts).mean()


def multipositive_nce(pocket_emb, ligand_emb, names, temperature):
    """Positives are every ligand with the same residue name, including itself."""
    logits = pocket_emb @ ligand_emb.T / temperature
    mask = same_name_mask(names, pocket_emb.device)
    return 0.5 * (_nce_rows(logits, mask) + _nce_rows(logits.T, mask))


class PocketLigandRetriever(nn.Module):
    def __init__(self, dim=64, temperature=0.07):
        super().__init__()
        self.pocket = GraphEncoder(25, out_dim=dim, d_max=10.0)
        self.ligand = GraphEncoder(6, out_dim=dim, d_max=2.5)
        self.temperature = temperature

    def encode_graphs(self, graphs, encoder, device):
        packed = _pack_graphs(graphs, device)
        return encoder(
            packed["x"],
            packed["edge_index"],
            packed["degree"],
            packed["edge_rbf"],
            packed["shape"],
            packed["batch"],
            packed["n_graphs"],
        )

    def loss(self, pocket_emb, ligand_emb, names):
        return multipositive_nce(pocket_emb, ligand_emb, names, self.temperature)


class CoordinateDenoiser(nn.Module):
    """Predict clean ligand coordinates from a noised copy and a pocket vector."""

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


def _pack_graphs(graphs, device):
    xs, degrees, rbfs, shapes, srcs, dsts, batches = [], [], [], [], [], [], []
    offset = 0
    for index, graph in enumerate(graphs):
        n_nodes = len(graph["x"])
        xs.append(torch.tensor(graph["x"], dtype=torch.float32, device=device))
        degrees.append(torch.tensor(graph["degree"], dtype=torch.float32, device=device))
        rbfs.append(torch.tensor(graph["edge_rbf"], dtype=torch.float32, device=device))
        shapes.append(torch.tensor(graph["shape"], dtype=torch.float32, device=device))
        edges = torch.tensor(graph["edge_index"], dtype=torch.long, device=device)
        srcs.append(edges[0] + offset)
        dsts.append(edges[1] + offset)
        batches.append(torch.full((n_nodes,), index, device=device, dtype=torch.long))
        offset += n_nodes
    return {
        "x": torch.cat(xs),
        "edge_index": torch.stack([torch.cat(srcs), torch.cat(dsts)]),
        "degree": torch.cat(degrees),
        "edge_rbf": torch.cat(rbfs),
        "shape": torch.stack(shapes),
        "batch": torch.cat(batches),
        "n_graphs": len(graphs),
    }


def retrieval_batch(model, complexes, device):
    attach_geometry(
        complexes,
        model.pocket.d_max,
        model.ligand.d_max,
        model.pocket.n_rbf,
        model.pocket.hist_bins,
    )
    pockets = model.encode_graphs([item["pocket"] for item in complexes], model.pocket, device)
    ligands = model.encode_graphs([item["ligand"] for item in complexes], model.ligand, device)
    return pockets, ligands


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
