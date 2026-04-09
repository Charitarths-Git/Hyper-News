from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_parquet_path(data_dir: Path) -> Path:
    for candidate in (data_dir / "articles.parquet", data_dir / "news_processed.parquet"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find articles.parquet or news_processed.parquet under {data_dir}"
    )


def generate_article_ids(data_dir: Path, output_path: Path, force: bool = False) -> Path:
    if output_path.exists() and not force:
        return output_path

    parquet_path = default_parquet_path(data_dir)
    dataframe = pd.read_parquet(parquet_path, columns=["news_id"])
    article_ids = [str(news_id).strip() for news_id in dataframe["news_id"].tolist() if str(news_id).strip()]

    embeddings_path = data_dir / "article_embeddings.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if len(article_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch between IDs ({len(article_ids)}) and embeddings ({len(embeddings)})."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(article_ids, handle)

    return output_path


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Generate article_ids.json aligned to article_embeddings.npy")
    parser.add_argument("--data-dir", default=str(root / "data"))
    parser.add_argument("--output", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else data_dir / "article_ids.json"
    written_path = generate_article_ids(data_dir=data_dir, output_path=output_path, force=args.force)
    print(f"article_ids.json ready -> {written_path}")


if __name__ == "__main__":
    main()
