from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str], cwd: Path) -> None:
    print(f"\n> {' '.join(command)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the exported HyperNews training bundles")
    parser.add_argument("--bundle-root", default=str(PROJECT_ROOT / "artifacts" / "training_bundle"))
    parser.add_argument("--h100-bundle-root", default=str(PROJECT_ROOT / "artifacts" / "training_bundle_h100_gnn"))
    parser.add_argument("--mind-path", default=str(PROJECT_ROOT / "data" / "mind_full" / "MIND-small"))
    parser.add_argument("--max-impressions", type=int, default=32)
    args = parser.parse_args()

    python_exe = sys.executable
    bundle_root = Path(args.bundle_root).resolve()
    h100_bundle_root = Path(args.h100_bundle_root).resolve()
    mind_path = Path(args.mind_path).resolve()

    if not bundle_root.exists():
        raise FileNotFoundError(f"Bundle root not found: {bundle_root}")
    if not h100_bundle_root.exists():
        raise FileNotFoundError(f"H100 bundle root not found: {h100_bundle_root}")
    if not mind_path.exists():
        raise FileNotFoundError(f"MIND path not found: {mind_path}")

    for root in (bundle_root, h100_bundle_root):
        if not (root / "hyperpersonalisedNewsReccomendation" / "data" / "article_ids.json").exists():
            raise FileNotFoundError(f"article_ids.json missing under bundle: {root}")

    run([python_exe, "training_packages/train_rtx4060.py", "--help"], cwd=bundle_root)
    run([python_exe, "training_packages/train_h100.py", "--help"], cwd=h100_bundle_root)

    smoke_root = PROJECT_ROOT / "artifacts" / "smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)

    run(
        [
            python_exe,
            "training_packages/train_rtx4060.py",
            "--mind-path",
            str(mind_path),
            "--data-dir",
            "hyperpersonalisedNewsReccomendation/data",
            "--output-dir",
            str(smoke_root / "rtx4060"),
            "--phase",
            "A",
            "--max-impressions",
            str(args.max_impressions),
        ],
        cwd=bundle_root,
    )

    run(
        [
            python_exe,
            "training_packages/train_h100.py",
            "--mind-path",
            str(mind_path),
            "--data-dir",
            "hyperpersonalisedNewsReccomendation/data",
            "--output-dir",
            str(smoke_root / "h100"),
            "--phase",
            "A",
            "--run-gnn",
            "0",
            "--max-impressions",
            str(args.max_impressions),
        ],
        cwd=h100_bundle_root,
    )

    for required in (
        smoke_root / "rtx4060" / "dqn_policy.pt",
        smoke_root / "rtx4060" / "eval_results.json",
        smoke_root / "h100" / "dqn_policy.pt",
        smoke_root / "h100" / "eval_results.json",
    ):
        if not required.exists():
            raise FileNotFoundError(f"Expected smoke output missing: {required}")

    print("\nBundle verification complete.")


if __name__ == "__main__":
    main()
