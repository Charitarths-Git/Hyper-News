"""
Prepare a slim runtime asset bundle for free cloud deployments.

Examples:
    python scripts/prepare_cloud_assets.py --limit 2000
    python scripts/prepare_cloud_assets.py --limit 2000 --in-place
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "articles.parquet"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data" / "article_embeddings.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cloud_runtime"
DEFAULT_LTR_MODEL = PROJECT_ROOT / "models" / "ltr_model.txt"
DEFAULT_LTR_WEIGHTS = PROJECT_ROOT / "models" / "ltr_model.txt.weights.json"


def _balanced_indices(df: pd.DataFrame, limit: int) -> np.ndarray:
    if limit <= 0 or len(df) <= limit:
        return df.index.to_numpy()

    if "category" not in df.columns:
        return df.head(limit).index.to_numpy()

    working = df.copy()
    if "popularity" in working.columns:
        working = working.sort_values("popularity", ascending=False)

    grouped: dict[str, list[int]] = {}
    category_priority: list[tuple[str, float]] = []
    category_series = working["category"].fillna("").astype(str).str.lower()
    for category, group in working.groupby(category_series, sort=False):
        rows = list(group.index)
        if not rows:
            continue
        grouped[category] = rows
        top_popularity = float(group["popularity"].iloc[0]) if "popularity" in group.columns else 0.0
        category_priority.append((category, top_popularity))

    ordered_categories = [category for category, _ in sorted(category_priority, key=lambda item: item[1], reverse=True)]
    if not ordered_categories:
        return working.head(limit).index.to_numpy()

    selected: list[int] = []
    cursors = {category: 0 for category in ordered_categories}

    while len(selected) < limit:
        progress = False
        for category in ordered_categories:
            rows = grouped[category]
            cursor = cursors[category]
            if cursor >= len(rows):
                continue
            selected.append(rows[cursor])
            cursors[category] += 1
            progress = True
            if len(selected) >= limit:
                break
        if not progress:
            break

    return np.asarray(selected, dtype=np.int64)


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    if source.resolve() == destination.resolve():
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def prepare_assets(parquet_path: Path, embeddings_path: Path, limit: int) -> tuple[pd.DataFrame, np.ndarray]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet source: {parquet_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embedding source: {embeddings_path}")

    dataframe = pd.read_parquet(parquet_path)
    embeddings = np.asarray(np.load(embeddings_path), dtype=np.float32)

    if len(dataframe) != len(embeddings):
        raise ValueError(
            f"Dataset and embedding count mismatch: {len(dataframe)} rows vs {len(embeddings)} vectors"
        )

    selected_idx = _balanced_indices(dataframe, limit)
    trimmed_df = dataframe.loc[selected_idx].reset_index(drop=True)
    trimmed_embeddings = np.asarray(embeddings[selected_idx], dtype=np.float32)
    return trimmed_df, trimmed_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare slim HyperNews runtime assets for cloud deployment.")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum number of articles to keep.")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET), help="Source parquet path.")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS), help="Source embeddings .npy path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output root for the staged bundle when not using --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the canonical data/ assets for a deploy branch.",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet).resolve()
    embeddings_path = Path(args.embeddings).resolve()
    trimmed_df, trimmed_embeddings = prepare_assets(parquet_path, embeddings_path, args.limit)

    if args.in_place:
        output_root = PROJECT_ROOT
    else:
        output_root = Path(args.output_dir).resolve()

    data_dir = output_root / "data"
    models_dir = output_root / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = data_dir / "articles.parquet"
    out_embeddings = data_dir / "article_embeddings.npy"

    trimmed_df.to_parquet(out_parquet, index=False)
    np.save(out_embeddings, trimmed_embeddings)

    copied_model = _copy_if_exists(DEFAULT_LTR_MODEL, models_dir / DEFAULT_LTR_MODEL.name)
    copied_weights = _copy_if_exists(DEFAULT_LTR_WEIGHTS, models_dir / DEFAULT_LTR_WEIGHTS.name)

    category_mix = (
        trimmed_df["category"].fillna("").astype(str).str.lower().value_counts().head(6).to_dict()
        if "category" in trimmed_df.columns
        else {}
    )

    print(f"Prepared {len(trimmed_df):,} runtime articles -> {out_parquet}")
    print(f"Saved embeddings {trimmed_embeddings.shape} -> {out_embeddings}")
    print(f"Top categories: {category_mix}")
    print(f"LTR model copied: {'yes' if copied_model else 'no'}")
    print(f"LTR fallback weights copied: {'yes' if copied_weights else 'no'}")


if __name__ == "__main__":
    main()
