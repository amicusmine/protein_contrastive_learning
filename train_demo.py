"""Train pocket–ligand retrieval and coordinate denoising with a held-out test set."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from retrieval_and_generation import (
    CoordinateDenoiser,
    PocketLigandRetriever,
    denoise_loss,
    kabsch_rmsd,
    retrieval_batch,
    sample_coords,
)
from structures import load_all

FIGURES = Path("figures")
MODELS = Path("models")


def exact_top1(pocket_emb, ligand_emb):
    similarity = pocket_emb @ ligand_emb.T
    top = similarity.argmax(dim=1)
    hits = top == torch.arange(top.shape[0], device=top.device)
    return float(hits.float().mean())


def full_gallery_rows(test_pockets, all_ligands, ordered, test_items):
    similarity = test_pockets @ all_ligands.T
    order = similarity.argsort(dim=1, descending=True)
    n_train = len(ordered) - len(test_items)
    rows = []
    for i, item in enumerate(test_items):
        true_index = n_train + i
        rank = int((order[i] == true_index).nonzero()[0]) + 1
        top = int(order[i, 0])
        guess = ordered[top]
        rows.append(
            {
                "pdb_id": item["pdb_id"],
                "ligand": item["ligand_name"],
                "protein": item["title"],
                "exact_rank": rank,
                "gallery_size": len(ordered),
                "top_hit": f"{guess['pdb_id']} {guess['ligand_name']}",
                "same_ligand": guess["ligand_name"] == item["ligand_name"],
            }
        )
    return similarity.cpu().numpy(), rows


def plot_losses(retrieval_losses, generation_losses, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].plot(np.arange(1, len(retrieval_losses) + 1), retrieval_losses, color="#1f4e79", lw=1.2)
    axes[0].set_title("Retrieval")
    axes[1].plot(np.arange(1, len(generation_losses) + 1), generation_losses, color="#b85c38", lw=1.2)
    axes[1].set_title("Coordinate denoising")
    for axis in axes:
        axis.set_xlabel("Step")
        axis.set_ylabel("Loss")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_similarity(similarity, ordered, test_items, path):
    labels = [f"{item['pdb_id']} {item['ligand_name']}" for item in ordered]
    test_labels = [f"{item['pdb_id']} {item['ligand_name']}" for item in test_items]
    n_train = len(ordered) - len(test_items)
    fig_w = max(8.5, 0.46 * len(labels) + 2.4)
    fig_h = 0.48 * len(test_labels) + 2.2
    fig, axis = plt.subplots(figsize=(fig_w, fig_h))
    image = axis.imshow(similarity, cmap="viridis", aspect="auto")
    axis.axvline(n_train - 0.5, color="white", lw=0.8)
    for row in range(len(test_items)):
        axis.add_patch(
            Rectangle((n_train + row - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="white", lw=1.6)
        )
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(test_labels)))
    axis.set_yticklabels(test_labels, fontsize=8)
    axis.set_xlabel("Ligand")
    axis.set_ylabel("Held-out pocket")
    axis.set_title("Cosine similarity. The white box is the true ligand.")
    fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rmsd(rows, path):
    fig, axis = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(rows))
    colors = ["#1f4e79" if row["split"] == "train" else "#c47b2b" for row in rows]
    axis.bar(x, [row["rmsd"] for row in rows], color=colors)
    test_mean = float(np.mean([row["rmsd"] for row in rows if row["split"] == "test"]))
    axis.axhline(test_mean, color="#c47b2b", ls="--", lw=1)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{row['pdb_id']} {row['ligand']}" for row in rows], rotation=55, ha="right", fontsize=8
    )
    axis.set_ylabel("Kabsch RMSD (Å)")
    axis.set_title("Denoised ligand coordinates")
    axis.legend(
        handles=[
            Patch(facecolor="#1f4e79", label="train"),
            Patch(facecolor="#c47b2b", label="test"),
            Line2D([0], [0], color="#c47b2b", ls="--", label=f"test mean {test_mean:.2f} Å"),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.55),
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-steps", type=int, default=800)
    parser.add_argument("--generation-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    complexes = load_all()
    train = [item for item in complexes if item["split"] == "train"]
    test = [item for item in complexes if item["split"] == "test"]
    if not train or not test:
        raise RuntimeError("both a train split and a test split are required")

    retriever = PocketLigandRetriever().to(device)
    optimizer = torch.optim.Adam(retriever.parameters(), lr=args.lr)
    retrieval_losses = []
    for step in range(1, args.retrieval_steps + 1):
        optimizer.zero_grad()
        pockets, ligands = retrieval_batch(retriever, train, device)
        loss = retriever.loss(pockets, ligands)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
        optimizer.step()
        retrieval_losses.append(float(loss.item()))
        if step == 1 or step % 50 == 0 or step == args.retrieval_steps:
            print(f"retrieval step {step}: loss {loss.item():.4f}")

    retriever.eval()
    ordered = train + test
    with torch.no_grad():
        all_pockets, all_ligands = retrieval_batch(retriever, ordered, device)
    n_train = len(train)
    train_top1 = exact_top1(all_pockets[:n_train], all_ligands[:n_train])
    test_gallery_top1 = exact_top1(all_pockets[n_train:], all_ligands[n_train:])
    similarity, test_rows = full_gallery_rows(
        all_pockets[n_train:], all_ligands, ordered, test
    )
    same_ligand_top1 = float(np.mean([row["same_ligand"] for row in test_rows]))
    print(f"train exact top-1, train ligands only: {train_top1:.2f}")
    print(f"test exact top-1, test ligands only: {test_gallery_top1:.2f}")
    print(f"test same-ligand top-1, all ligands: {same_ligand_top1:.2f}")
    for row in test_rows:
        print(
            f"{row['pdb_id']} {row['ligand']}: "
            f"exact rank {row['exact_rank']}/{row['gallery_size']}, "
            f"top hit {row['top_hit']}"
        )

    denoiser = CoordinateDenoiser().to(device)
    gen_opt = torch.optim.Adam(denoiser.parameters(), lr=args.lr)
    generation_losses = []
    for step in range(1, args.generation_steps + 1):
        index = step % n_train
        item = train[index]
        gen_opt.zero_grad()
        loss = denoise_loss(denoiser, all_pockets[index].detach(), item["ligand"], device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        gen_opt.step()
        generation_losses.append(float(loss.item()))
        if step == 1 or step % 100 == 0 or step == args.generation_steps:
            print(f"generation step {step}: loss {loss.item():.4f}")

    denoiser.eval()
    rmsd_rows = []
    for index, item in enumerate(ordered):
        pred = sample_coords(denoiser, all_pockets[index].detach(), item["ligand"], device)
        rmsd = kabsch_rmsd(pred, item["ligand"]["coords"])
        rmsd_rows.append(
            {
                "pdb_id": item["pdb_id"],
                "ligand": item["ligand_name"],
                "split": item["split"],
                "rmsd": rmsd,
            }
        )
        print(f"{item['split']} {item['pdb_id']} {item['ligand_name']}: denoised RMSD {rmsd:.2f} A")
    train_rmsd = float(np.mean([row["rmsd"] for row in rmsd_rows if row["split"] == "train"]))
    test_rmsd = float(np.mean([row["rmsd"] for row in rmsd_rows if row["split"] == "test"]))
    print(f"mean denoised RMSD train {train_rmsd:.2f} A, test {test_rmsd:.2f} A")

    FIGURES.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    plot_losses(retrieval_losses, generation_losses, FIGURES / "loss.png")
    plot_similarity(similarity, ordered, test, FIGURES / "retrieval_similarity.png")
    plot_rmsd(rmsd_rows, FIGURES / "denoising_rmsd.png")

    torch.save(
        {
            "state_dict": retriever.state_dict(),
            "dim": 64,
            "temperature": retriever.temperature,
            "train_ids": [item["pdb_id"] for item in train],
            "test_ids": [item["pdb_id"] for item in test],
            "seed": args.seed,
            "retrieval_steps": args.retrieval_steps,
            "lr": args.lr,
        },
        MODELS / "retriever.pt",
    )
    torch.save(
        {
            "state_dict": denoiser.state_dict(),
            "pocket_dim": 64,
            "hidden": 128,
            "seed": args.seed,
            "generation_steps": args.generation_steps,
            "lr": args.lr,
        },
        MODELS / "denoiser.pt",
    )

    metrics = {
        "seed": args.seed,
        "retrieval_steps": args.retrieval_steps,
        "generation_steps": args.generation_steps,
        "lr": args.lr,
        "device": str(device),
        "train_exact_top1": train_top1,
        "test_exact_top1_in_test_gallery": test_gallery_top1,
        "test_same_ligand_top1_in_full_gallery": same_ligand_top1,
        "mean_rmsd_train": train_rmsd,
        "mean_rmsd_test": test_rmsd,
        "test_retrieval": test_rows,
        "rmsd": rmsd_rows,
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (Path("data/processed") / "heldout_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pdb_id", "ligand", "protein", "exact_rank", "gallery_size", "top_hit", "same_ligand"],
        )
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"wrote {FIGURES} and {MODELS}")


if __name__ == "__main__":
    main()
