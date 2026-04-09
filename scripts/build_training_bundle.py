from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from generate_article_ids import generate_article_ids


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_SOURCE = PROJECT_ROOT / "training_packages"
BACKEND_SOURCE = PROJECT_ROOT / "backend"
DATA_SOURCE = PROJECT_ROOT / "data"
GRAPH_SOURCE = PROJECT_ROOT / "graph"

TRAINING_FILES = [
    "README_TRAINING.md",
    "requirements_train.txt",
    "path_utils.py",
    "train_bio_encoder.py",
    "train_colab_t4.py",
    "train_dqn_core.py",
    "train_h100.py",
    "train_kg_gnn.py",
    "train_rtx3050.py",
    "train_rtx4060.py",
]

BACKEND_FILES = [
    Path("logging_policy.py"),
    Path("models") / "bio_encoder.py",
    Path("models") / "dqn_policy.py",
    Path("models") / "history_encoder.py",
    Path("models") / "kg_gnn.py",
]

DATA_FILES = [
    "article_embeddings.npy",
    "article_ids.json",
]

OPTIONAL_GRAPH_FILE = "knowledge_graph.pkl"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_manifest(bundle_root: Path, include_graph: bool) -> None:
    manifest = {
        "bundle_root": str(bundle_root),
        "project_root": str(PROJECT_ROOT),
        "include_graph": include_graph,
        "training_files": TRAINING_FILES,
        "backend_files": [str(path).replace("\\", "/") for path in BACKEND_FILES],
        "data_files": DATA_FILES,
        "optional_graph_file": OPTIONAL_GRAPH_FILE if include_graph else "",
        "datasets_to_upload": {
            "rtx4060_default": "MIND-small train + dev",
            "rtx4060_optional": "MIND-large capped with --max-impressions 150000..200000",
            "h100_default": "MIND-large train + dev",
        },
        "artifacts_to_bring_back": [
            "bio_encoder.pt",
            "dqn_policy.pt",
            "logging_policy.pkl",
            "eval_results.json",
            "metrics.jsonl",
            "kg_gnn.pt",
            "kg_embeddings.npy",
        ],
        "do_not_move": [
            "frontend/",
            ".next/",
            "node_modules/",
            "data/faiss_mind.index",
            "data/news_faiss.index",
            "data/qdrant_local_py312/",
            "data/hypernews.db",
            "data/ltr_features*.csv",
            "*.log",
            "docker-compose.yml",
            ".git/",
        ],
    }
    with (bundle_root / "BUNDLE_MANIFEST.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def build_bundle(bundle_root: Path, include_graph: bool) -> Path:
    generate_article_ids(DATA_SOURCE, DATA_SOURCE / "article_ids.json")
    reset_dir(bundle_root)

    training_target = bundle_root / "training_packages"
    project_target = bundle_root / "hyperpersonalisedNewsReccomendation"
    backend_target = project_target / "backend"
    data_target = project_target / "data"
    graph_target = project_target / "graph"

    training_target.mkdir(parents=True, exist_ok=True)
    backend_target.mkdir(parents=True, exist_ok=True)
    data_target.mkdir(parents=True, exist_ok=True)
    graph_target.mkdir(parents=True, exist_ok=True)

    for name in TRAINING_FILES:
        source = TRAINING_SOURCE / name
        if not source.exists():
            raise FileNotFoundError(f"Missing training source file: {source}")
        copy_file(source, training_target / name)

    for relative_path in BACKEND_FILES:
        source = BACKEND_SOURCE / relative_path
        if not source.exists():
            raise FileNotFoundError(f"Missing backend source file: {source}")
        copy_file(source, backend_target / relative_path)

    for name in DATA_FILES:
        source = DATA_SOURCE / name
        if not source.exists():
            raise FileNotFoundError(f"Missing data file: {source}")
        copy_file(source, data_target / name)

    if include_graph:
        graph_source = GRAPH_SOURCE / OPTIONAL_GRAPH_FILE
        if not graph_source.exists():
            raise FileNotFoundError(f"Missing graph file: {graph_source}")
        copy_file(graph_source, graph_target / OPTIONAL_GRAPH_FILE)

    write_manifest(bundle_root, include_graph=include_graph)
    return bundle_root


def zip_bundle(bundle_root: Path, output_zip: Path) -> Path:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_root.parent))
    return output_zip


def export_zip(source_zip: Path, downloads_dir: Path | None) -> Path | None:
    if downloads_dir is None:
        return None
    downloads_dir.mkdir(parents=True, exist_ok=True)
    target = downloads_dir / source_zip.name
    shutil.copy2(source_zip, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portable HyperNews training bundles for 4060 and H100")
    parser.add_argument("--artifacts-dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--downloads-dir", default=str(Path.home() / "Downloads"))
    parser.add_argument("--skip-download-export", action="store_true")
    parser.add_argument("--skip-zips", action="store_true")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    downloads_dir = None if args.skip_download_export else Path(args.downloads_dir).resolve()

    base_bundle = build_bundle(artifacts_dir / "training_bundle", include_graph=False)
    h100_bundle = build_bundle(artifacts_dir / "training_bundle_h100_gnn", include_graph=True)

    print(f"Base bundle ready -> {base_bundle}")
    print(f"H100 bundle ready -> {h100_bundle}")

    if args.skip_zips:
        return

    base_zip = zip_bundle(base_bundle, artifacts_dir / "zips" / "hypernews_training_bundle.zip")
    h100_zip = zip_bundle(h100_bundle, artifacts_dir / "zips" / "hypernews_training_bundle_h100_gnn.zip")

    base_export = export_zip(base_zip, downloads_dir)
    h100_export = export_zip(h100_zip, downloads_dir)

    print(f"Base zip ready -> {base_zip}")
    print(f"H100 zip ready -> {h100_zip}")
    if base_export:
        print(f"Copied base zip to -> {base_export}")
    if h100_export:
        print(f"Copied H100 zip to -> {h100_export}")


if __name__ == "__main__":
    main()
