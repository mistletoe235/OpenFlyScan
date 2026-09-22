"""Predict normalized regional GS difference from prepared feature arrays."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openflyscan.inference import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("--output must end in .npy")
    if args.output.exists():
        parser.error("Output exists; choose another path")
    scores = predict(args.checkpoint, args.inputs, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, scores, allow_pickle=False)
    print(f"Wrote {len(scores)} regional scores to {args.output}")


if __name__ == "__main__":
    main()
