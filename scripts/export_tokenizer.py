import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from tokenizers import Tokenizer

from nlp_codegen.data import validate_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("Output must differ from the checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("tokenizer_json"), str):
        raise ValueError("Checkpoint must contain tokenizer_json")
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
    validate_tokenizer(tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(checkpoint["tokenizer_json"], encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
