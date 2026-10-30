"""Train only the optional geometry routing head from offline token caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .routing_head import (
    GRMGeometryRoutingHead,
    GRMGeometryRoutingOutput,
    GRMGeometryRoutingTargets,
    geometry_routing_loss,
)


class RoutingCacheDataset(Dataset):
    """Read a ``torch.save`` list of per-pair routing training dictionaries."""

    def __init__(self, path: Path) -> None:
        records = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(records, list) or not records:
            raise ValueError("Routing cache must be a non-empty list of dictionaries.")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return self.records[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = RoutingCacheDataset(args.cache)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
    head = GRMGeometryRoutingHead(args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    head.train()
    for epoch in range(args.epochs):
        totals = []
        for record in loader:
            tokens = record["selected_graph_tokens"].to(device)
            valid = record["valid_mask"].to(device)
            targets = GRMGeometryRoutingTargets(
                target_h=record["target_h"].to(device),
                target_deform=record["target_deform"].to(device),
                h_mask=record["h_mask"].to(device),
                deform_mask=record["deform_mask"].to(device),
            )
            output = head(tokens, valid)
            losses = geometry_routing_loss(output, targets)
            optimizer.zero_grad(set_to_none=True)
            losses["routing_total"].backward()
            optimizer.step()
            totals.append(float(losses["routing_total"].detach()))
        print(f"epoch={epoch + 1} routing_total={sum(totals) / max(len(totals), 1):.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "routing_head": head.state_dict(),
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
