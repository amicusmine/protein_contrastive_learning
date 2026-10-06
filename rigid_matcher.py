"""Train one rigid motion that places a ligand in a pocket, then rank ligands by the contacts of that pose.

The pocket and the ligand are written in their own local frames. The network predicts
a rotation and a translation. A second network scores the placed ligand.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from retrieval_and_generation import _nce_rows, same_name_mask
from structures import _pocket_frame


def intra_hist(coords, bins=8, d_max=12.0):
    coords = np.asarray(coords, dtype=np.float32)
    hist = np.zeros((len(coords), bins), dtype=np.float32)
    if len(coords) < 2:
        return hist
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    centers = np.linspace(0, d_max, bins, dtype=np.float32)
    width = np.float32(d_max / max(bins - 1, 1))
    eye = np.eye(len(coords), dtype=bool)
    for index in range(len(coords)):
        values = dist[index, ~eye[index]]
        weights = np.exp(-0.5 * ((values[:, None] - centers) / width) ** 2)
        total = float(weights.sum())
        if total > 0:
            hist[index] = weights.sum(axis=0) / total
    return hist


def prepare_geometry(complexes):
    for item in complexes:
        ligand = item["ligand"]
        if "canonical" not in ligand:
            center, axes = _pocket_frame(ligand["coords_abs"])
            ligand["canonical"] = (ligand["coords_abs"] - center) @ axes.T
            ligand["intra"] = intra_hist(ligand["canonical"])
    return complexes


def _tensor(array, device):
    return torch.tensor(np.asarray(array), dtype=torch.float32, device=device)


def pack_complex(item, device):
    features = item["pocket"]["x"].copy()
    features[:, -1] = features[:, -1] / 100.0
    ligand = item["ligand"]
    return {
        "pocket_x": _tensor(features, device),
        "pocket_xyz": _tensor(item["pocket"]["frame_coords"], device),
        "lig_x": _tensor(ligand["x"], device),
        "intra": _tensor(ligand["intra"], device),
        "canonical": _tensor(ligand["canonical"], device),
        "bound": _tensor(ligand["bound_coords"], device),
    }


def rotation_from_6d(raw):
    a1 = raw[..., 0:3]
    a2 = raw[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def point_rmsd(pred, true):
    return torch.sqrt(((pred - true) ** 2).sum(dim=-1).mean())


class RigidMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.residue = nn.Sequential(
            nn.Linear(25 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.atom = nn.Sequential(
            nn.Linear(6 + 8 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.pose = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        )
        self.score = nn.Sequential(
            nn.Linear(6 * 25 + 6 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.pose[-1].weight)
        nn.init.zeros_(self.pose[-1].bias)
        self.pose[-1].bias.data[:6] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def pocket_summary(self, pocket_x, pocket_xyz):
        hidden = self.residue(torch.cat([pocket_x, pocket_xyz / 10.0], dim=-1))
        return torch.cat([hidden.mean(dim=0), hidden.max(dim=0).values], dim=0)

    def ligand_summary(self, ligand):
        hidden = self.atom(
            torch.cat([ligand["lig_x"], ligand["intra"], ligand["canonical"] / 10.0], dim=-1)
        )
        return torch.cat([hidden.mean(dim=0), hidden.max(dim=0).values], dim=0)

    def place(self, pocket, ligand):
        summary = torch.cat(
            [
                self.pocket_summary(pocket["pocket_x"], pocket["pocket_xyz"]),
                self.ligand_summary(ligand),
            ],
            dim=0,
        )
        raw = self.pose(summary)
        rotation = rotation_from_6d(raw[:6])
        translation = raw[6:] * 10.0
        return ligand["canonical"] @ rotation.T + translation

    def contact_features(self, lig_xyz, lig_elem, pocket_xyz, pocket_x):
        dist = torch.cdist(lig_xyz, pocket_xyz)
        shell = torch.exp(-0.5 * ((dist - 4.5) / 1.3) ** 2)
        shell = shell / shell.sum().clamp(min=1e-6)
        pair = torch.einsum("ae,ap,pr->er", lig_elem, shell, pocket_x)
        pair = pair / pair.sum().clamp(min=1)
        nearest = dist.min(dim=1).values
        centers = torch.linspace(0, 12.0, 12, device=lig_xyz.device, dtype=lig_xyz.dtype)
        width = 12.0 / 11.0
        weights = torch.exp(-0.5 * ((nearest.unsqueeze(-1) - centers) / width) ** 2)
        hist = lig_elem.T @ weights
        hist = hist / hist.sum().clamp(min=1)
        return torch.cat([pair.reshape(-1), hist.reshape(-1)], dim=0)


def train_pose(model, packed, epochs, lr, test_packed=None, augment=False):
    parameters = list(model.residue.parameters()) + list(model.atom.parameters()) + list(model.pose.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(packed))
        total = 0.0
        for index in order:
            item = packed[int(index)]
            optimizer.zero_grad()
            pocket_xyz = item["pocket_xyz"]
            target = item["bound"]
            if augment:
                rotation = rotation_from_6d(torch.randn(6, device=target.device))
                pocket_xyz = pocket_xyz @ rotation.T
                target = target @ rotation.T
            pocket = dict(item)
            pocket["pocket_xyz"] = pocket_xyz
            loss = F.mse_loss(model.place(pocket, item), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.item())
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            train_rmsd = placement_rmsd(model, packed)
            message = f"rigid pose epoch {epoch}: mse {total / len(packed):.4f}, train RMSD {train_rmsd:.3f} A"
            if test_packed is not None:
                message += f", test RMSD {placement_rmsd(model, test_packed):.3f} A"
            print(message, flush=True)


@torch.no_grad()
def placement_rmsd(model, packed):
    was_training = model.training
    model.eval()
    values = [float(point_rmsd(model.place(item, item), item["bound"])) for item in packed]
    model.train(was_training)
    return float(np.mean(values))


def _contact_matrix(model, queries, gallery):
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for pocket in queries:
            row = []
            for ligand in gallery:
                placed = model.place(pocket, ligand)
                row.append(
                    model.contact_features(placed, ligand["lig_x"], pocket["pocket_xyz"], pocket["pocket_x"])
                )
            rows.append(torch.stack(row))
    model.train(was_training)
    return torch.stack(rows)


def _same_count(scores, names):
    top = scores.argmax(dim=1).tolist()
    exact = sum(index == choice for index, choice in enumerate(top))
    same = sum(names[choice] == names[index] for index, choice in enumerate(top))
    return exact, same


def train_score(model, queries, names, epochs, lr):
    features = _contact_matrix(model, queries, queries).detach()
    optimizer = torch.optim.Adam(model.score.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        flat = features.reshape(-1, features.shape[-1])
        scores = model.score(flat).reshape(features.shape[0], features.shape[1])
        mask = same_name_mask(names, scores.device)
        loss = 0.5 * (_nce_rows(scores, mask) + _nce_rows(scores.T, mask))
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            exact, same = _same_count(scores.detach(), names)
            print(
                f"rigid score epoch {epoch}: loss {loss.item():.4f}, "
                f"exact {exact}/{len(names)}, same ligand {same}/{len(names)}",
                flush=True,
            )


@torch.no_grad()
def rigid_score_matrix(model, queries, gallery):
    model.eval()
    features = _contact_matrix(model, queries, gallery)
    scores = model.score(features.reshape(-1, features.shape[-1])).reshape(len(queries), len(gallery))
    return scores.detach().cpu().numpy()


@torch.no_grad()
def true_ligand_rmsd(model, packed):
    model.eval()
    return [float(point_rmsd(model.place(item, item), item["bound"])) for item in packed]
