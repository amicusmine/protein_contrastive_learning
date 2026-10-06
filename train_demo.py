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
from rigid_matcher import (
    RigidMatcher,
    pack_complex,
    prepare_geometry,
    rigid_score_matrix,
    train_pose,
    train_score,
    true_ligand_rmsd,
)
from structures import load_all

FIGURES = Path("figures")
MODELS = Path("models")


def retrieval_metrics(scores, queries, gallery):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, axis=1)
    ids = [item["pdb_id"] for item in gallery]
    names = [item["ligand_name"] for item in gallery]
    exact_hit, same_hit, ranks, rows = [], [], [], []
    for i, item in enumerate(queries):
        exact_index = ids.index(item["pdb_id"])
        rank = int(np.where(order[i] == exact_index)[0][0]) + 1
        top = int(order[i, 0])
        guess = gallery[top]
        exact_hit.append(rank == 1)
        same_hit.append(names[top] == item["ligand_name"])
        ranks.append(rank)
        rows.append(
            {
                "pdb_id": item["pdb_id"],
                "ligand": item["ligand_name"],
                "protein": item["title"],
                "exact_rank": rank,
                "gallery_size": len(gallery),
                "top_hit": f"{guess['pdb_id']} {guess['ligand_name']}",
                "same_ligand": names[top] == item["ligand_name"],
            }
        )
    return {
        "exact_top1": float(np.mean(exact_hit)) if exact_hit else 0.0,
        "same_ligand_top1": float(np.mean(same_hit)) if same_hit else 0.0,
        "mean_exact_rank": float(np.mean(ranks)) if ranks else 0.0,
        "n_exact": int(np.sum(exact_hit)),
        "n_same": int(np.sum(same_hit)),
        "n": len(queries),
        "rows": rows,
    }


def chance_same_ligand(queries, gallery):
    names = [item["ligand_name"] for item in gallery]
    if not names:
        return 0.0
    return float(np.mean([names.count(item["ligand_name"]) / len(names) for item in queries]))


def moving_mean(values, window):
    values = np.asarray(values, dtype=float)
    window = max(1, min(int(window), len(values)))
    if window == 1:
        return values
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_losses(retrieval_losses, generation_losses, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    series = (
        (retrieval_losses, "Retrieval", "#1f4e79", 25),
        (generation_losses, "Coordinate denoising", "#b85c38", 60),
    )
    for axis, (values, title, color, window) in zip(axes, series):
        steps = np.arange(1, len(values) + 1)
        axis.plot(steps, values, color=color, lw=0.7, alpha=0.28)
        axis.plot(steps, moving_mean(values, window), color=color, lw=1.8, label=f"{window}-step mean")
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel("Loss")
        axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_similarity(similarity, ordered, test_items, path):
    labels = [f"{item['pdb_id']} {item['ligand_name']}" for item in ordered]
    test_labels = [f"{item['pdb_id']} {item['ligand_name']}" for item in test_items]
    n_train = len(ordered) - len(test_items)
    fig_w = max(8.5, 0.28 * len(labels) + 2.2)
    fig_h = max(4.2, 0.32 * len(test_labels) + 1.8)
    fig, axis = plt.subplots(figsize=(fig_w, fig_h))
    image = axis.imshow(similarity, cmap="viridis", aspect="auto")
    axis.axvline(n_train - 0.5, color="white", lw=0.8)
    for row in range(len(test_items)):
        axis.add_patch(
            Rectangle((n_train + row - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="white", lw=1.6)
        )
    axis.set_xticks(np.arange(len(labels)))
    label_size = 6 if len(labels) > 40 else 8
    axis.set_xticklabels(labels, rotation=90, ha="center", fontsize=label_size)
    axis.set_yticks(np.arange(len(test_labels)))
    axis.set_yticklabels(test_labels, fontsize=label_size)
    axis.set_xlabel("Ligand")
    axis.set_ylabel("Held-out pocket")
    axis.set_title("Cosine similarity. The white box is the true ligand.")
    fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_comparison(learned, rigid, chance, path):
    labels = ["Train exact", "Train same ligand", "Held-out exact", "Held-out same ligand"]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 4.0))
    axis.bar(x - width / 2, learned, width, color="#1f4e79", label="Contrastive")
    axis.bar(x + width / 2, rigid, width, color="#c47b2b", label="Rigid-body fit")
    axis.scatter(x, chance, color="black", s=28, zorder=3, label="Random")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Top-1")
    axis.set_title("Retrieval against the same galleries")
    axis.legend(frameon=False, loc="upper right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rmsd(rows, path, title="Denoised ligand coordinates"):
    fig_w = max(9.2, 0.24 * len(rows) + 1.8)
    fig, axis = plt.subplots(figsize=(fig_w, 4.4))
    x = np.arange(len(rows))
    colors = ["#1f4e79" if row["split"] == "train" else "#c47b2b" for row in rows]
    axis.bar(x, [row["rmsd"] for row in rows], color=colors)
    test_mean = float(np.mean([row["rmsd"] for row in rows if row["split"] == "test"]))
    axis.axhline(test_mean, color="#c47b2b", ls="--", lw=1)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{row['pdb_id']} {row['ligand']}" for row in rows],
        rotation=90,
        ha="center",
        fontsize=7 if len(rows) > 30 else 8,
    )
    axis.set_ylabel("Kabsch RMSD (Å)")
    axis.set_title(title)
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
    parser.add_argument("--generation-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rigid-epochs", type=int, default=100)
    parser.add_argument("--rigid-score-epochs", type=int, default=240)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    complexes = load_all()
    train = [item for item in complexes if item["split"] == "train"]
    test = [item for item in complexes if item["split"] == "test"]
    if not train or not test:
        raise RuntimeError("both a train split and a test split are required")
    if args.generation_steps <= 0:
        args.generation_steps = 12 * len(train)
    print(
        f"{len(train)} train complexes, {len(test)} test complexes, "
        f"{args.generation_steps} denoising steps"
    )
    ordered = train + test
    n_train = len(train)
    train_names = [item["ligand_name"] for item in train]
    prepare_geometry(complexes)
    train_packed = [pack_complex(item, device) for item in train]
    test_packed = [pack_complex(item, device) for item in test]
    ordered_packed = train_packed + test_packed
    rigid = RigidMatcher().to(device)
    train_pose(
        rigid,
        train_packed,
        epochs=args.rigid_epochs,
        lr=1e-3,
        test_packed=test_packed,
    )
    train_score(rigid, train_packed, train_names, epochs=args.rigid_score_epochs, lr=3e-3)
    rigid_scores = rigid_score_matrix(rigid, ordered_packed, ordered_packed)
    rigid_pose = true_ligand_rmsd(rigid, ordered_packed)
    rigid_train = retrieval_metrics(rigid_scores[:n_train, :n_train], train, train)
    rigid_test_gallery = retrieval_metrics(rigid_scores[n_train:, n_train:], test, test)
    rigid_full = retrieval_metrics(rigid_scores[n_train:], test, ordered)
    print(
        "rigid train exact "
        f"{rigid_train['n_exact']}/{rigid_train['n']}, "
        f"same ligand {rigid_train['n_same']}/{rigid_train['n']}"
    )
    print(
        "rigid held-out exact "
        f"{rigid_test_gallery['n_exact']}/{rigid_test_gallery['n']}, "
        f"same ligand in full gallery {rigid_full['n_same']}/{rigid_full['n']}, "
        f"mean rank {rigid_full['mean_exact_rank']:.1f}, "
        f"pose RMSD train {float(np.mean(rigid_pose[:n_train])):.2f} A, "
        f"test {float(np.mean(rigid_pose[n_train:])):.2f} A"
    )

    retriever = PocketLigandRetriever().to(device)
    optimizer = torch.optim.Adam(retriever.parameters(), lr=args.lr)
    retrieval_losses = []
    for step in range(1, args.retrieval_steps + 1):
        optimizer.zero_grad()
        pockets, ligands = retrieval_batch(retriever, train, device)
        loss = retriever.loss(pockets, ligands, train_names)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
        optimizer.step()
        retrieval_losses.append(float(loss.item()))
        if step == 1 or step % 50 == 0 or step == args.retrieval_steps:
            print(f"retrieval step {step}: loss {loss.item():.4f}")

    retriever.eval()
    with torch.no_grad():
        all_pockets, all_ligands = retrieval_batch(retriever, ordered, device)
    similarity = (all_pockets @ all_ligands.T).cpu().numpy()
    learned_train = retrieval_metrics(similarity[:n_train, :n_train], train, train)
    learned_test_gallery = retrieval_metrics(similarity[n_train:, n_train:], test, test)
    learned_full = retrieval_metrics(similarity[n_train:], test, ordered)
    print(
        "contrastive train exact "
        f"{learned_train['n_exact']}/{learned_train['n']}, "
        f"same ligand {learned_train['n_same']}/{learned_train['n']}"
    )
    print(
        "contrastive held-out exact "
        f"{learned_test_gallery['n_exact']}/{learned_test_gallery['n']}, "
        f"same ligand in full gallery {learned_full['n_same']}/{learned_full['n']}, "
        f"mean rank {learned_full['mean_exact_rank']:.1f}"
    )
    for learned_row, rigid_row in zip(learned_full["rows"], rigid_full["rows"]):
        print(
            f"{learned_row['pdb_id']} {learned_row['ligand']}: "
            f"contrastive rank {learned_row['exact_rank']}/{learned_row['gallery_size']} "
            f"top {learned_row['top_hit']}; "
            f"rigid rank {rigid_row['exact_rank']} top {rigid_row['top_hit']}"
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
    chance = [
        1 / len(train),
        chance_same_ligand(train, train),
        1 / len(test),
        chance_same_ligand(test, ordered),
    ]
    learned_rates = [
        learned_train["exact_top1"],
        learned_train["same_ligand_top1"],
        learned_test_gallery["exact_top1"],
        learned_full["same_ligand_top1"],
    ]
    rigid_rates = [
        rigid_train["exact_top1"],
        rigid_train["same_ligand_top1"],
        rigid_test_gallery["exact_top1"],
        rigid_full["same_ligand_top1"],
    ]
    rigid_pose_by_id = {item["pdb_id"]: float(value) for item, value in zip(ordered, rigid_pose)}
    rmsd_by_id = {row["pdb_id"]: row["rmsd"] for row in rmsd_rows}
    heldout_rows = []
    for learned_row, rigid_row in zip(learned_full["rows"], rigid_full["rows"]):
        heldout_rows.append(
            {
                "pdb_id": learned_row["pdb_id"],
                "ligand": learned_row["ligand"],
                "protein": learned_row["protein"],
                "contrastive_rank": learned_row["exact_rank"],
                "contrastive_top_hit": learned_row["top_hit"],
                "contrastive_same_ligand": learned_row["same_ligand"],
                "rigid_rank": rigid_row["exact_rank"],
                "rigid_top_hit": rigid_row["top_hit"],
                "rigid_same_ligand": rigid_row["same_ligand"],
                "gallery_size": learned_row["gallery_size"],
                "rigid_rmsd": f"{rigid_pose_by_id[learned_row['pdb_id']]:.2f}",
                "rmsd": f"{rmsd_by_id[learned_row['pdb_id']]:.2f}",
            }
        )
    rigid_rmsd_rows = [
        {
            "pdb_id": item["pdb_id"],
            "ligand": item["ligand_name"],
            "split": item["split"],
            "rmsd": float(value),
        }
        for item, value in zip(ordered, rigid_pose)
    ]
    plot_losses(retrieval_losses, generation_losses, FIGURES / "loss.png")
    plot_similarity(similarity[n_train:], ordered, test, FIGURES / "retrieval_similarity.png")
    plot_comparison(learned_rates, rigid_rates, chance, FIGURES / "retrieval_comparison.png")
    plot_rmsd(rigid_rmsd_rows, FIGURES / "rigid_rmsd.png", title="Rigid placement of the true ligand")
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
            "positives": "same residue name",
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
    torch.save(
        {
            "state_dict": rigid.state_dict(),
            "seed": args.seed,
            "pose_epochs": args.rigid_epochs,
            "score_epochs": args.rigid_score_epochs,
        },
        MODELS / "rigid.pt",
    )

    metrics = {
        "seed": args.seed,
        "retrieval_steps": args.retrieval_steps,
        "generation_steps": args.generation_steps,
        "lr": args.lr,
        "device": str(device),
        "positives": "same residue name",
        "contrastive": {
            "train_exact_top1": learned_train["exact_top1"],
            "train_same_ligand_top1": learned_train["same_ligand_top1"],
            "test_exact_top1_in_test_gallery": learned_test_gallery["exact_top1"],
            "test_same_ligand_top1_in_full_gallery": learned_full["same_ligand_top1"],
            "test_mean_exact_rank": learned_full["mean_exact_rank"],
        },
        "rigid": {
            "train_exact_top1": rigid_train["exact_top1"],
            "train_same_ligand_top1": rigid_train["same_ligand_top1"],
            "test_exact_top1_in_test_gallery": rigid_test_gallery["exact_top1"],
            "test_same_ligand_top1_in_full_gallery": rigid_full["same_ligand_top1"],
            "test_mean_exact_rank": rigid_full["mean_exact_rank"],
            "pose_epochs": args.rigid_epochs,
            "score_epochs": args.rigid_score_epochs,
            "mean_pose_rmsd_train": float(np.mean(rigid_pose[:n_train])),
            "mean_pose_rmsd_test": float(np.mean(rigid_pose[n_train:])),
        },
        "chance": {
            "train_exact": chance[0],
            "train_same_ligand": chance[1],
            "test_exact": chance[2],
            "test_same_ligand_full_gallery": chance[3],
        },
        "mean_rmsd_train": train_rmsd,
        "mean_rmsd_test": test_rmsd,
        "test_retrieval": heldout_rows,
        "rmsd": rmsd_rows,
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (Path("data/processed") / "heldout_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(heldout_rows[0].keys()))
        writer.writeheader()
        writer.writerows(heldout_rows)
    print(f"wrote {FIGURES} and {MODELS}")


if __name__ == "__main__":
    main()
