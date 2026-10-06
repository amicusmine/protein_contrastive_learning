"""Train the toy retrieval and generation models on eight public complexes."""

import argparse

import numpy as np
import torch

from retrieval_and_generation import (
    CoordinateDenoiser,
    PocketLigandRetriever,
    denoise_loss,
    kabsch_rmsd,
    ranking_table,
    retrieval_batch,
    sample_coords,
)
from structures import load_all


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

    retriever = PocketLigandRetriever().to(device)
    optimizer = torch.optim.Adam(retriever.parameters(), lr=args.lr)
    for step in range(1, args.retrieval_steps + 1):
        optimizer.zero_grad()
        pockets, ligands = retrieval_batch(retriever, complexes, device)
        loss = retriever.loss(pockets, ligands)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.retrieval_steps:
            print(f"retrieval step {step}: loss {loss.item():.4f}")

    retriever.eval()
    with torch.no_grad():
        pockets, ligands = retrieval_batch(retriever, complexes, device)
    top1, lines = ranking_table(pockets, ligands, complexes)
    print(f"retrieval top-1 on the training set: {top1:.2f}")
    print("\n".join(lines))

    denoiser = CoordinateDenoiser().to(device)
    gen_opt = torch.optim.Adam(denoiser.parameters(), lr=args.lr)
    for step in range(1, args.generation_steps + 1):
        item = complexes[step % len(complexes)]
        index = step % len(complexes)
        gen_opt.zero_grad()
        loss = denoise_loss(denoiser, pockets[index].detach(), item["ligand"], device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        gen_opt.step()
        if step == 1 or step % 100 == 0 or step == args.generation_steps:
            print(f"generation step {step}: loss {loss.item():.4f}")

    denoiser.eval()
    rmsds = []
    for index, item in enumerate(complexes):
        pred = sample_coords(denoiser, pockets[index].detach(), item["ligand"], device)
        true = item["ligand"]["coords"]
        rmsd = kabsch_rmsd(pred, true)
        rmsds.append(rmsd)
        print(f"{item['pdb_id']} {item['ligand_name']}: denoised RMSD {rmsd:.2f} A")
    print(f"mean denoised RMSD: {float(np.mean(rmsds)):.2f} A")


if __name__ == "__main__":
    main()
