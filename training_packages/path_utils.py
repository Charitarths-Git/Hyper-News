from __future__ import annotations

from pathlib import Path


def resolve_project_root(start_file: str | Path) -> Path:
    """Resolve the HyperNews project root from either repo or bundle layout."""
    script_path = Path(start_file).resolve()
    base_dir = script_path.parent.parent

    candidates = [
        base_dir,
        base_dir / "hyperpersonalisedNewsReccomendation",
    ]

    for candidate in candidates:
        if (candidate / "backend").exists() and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve HyperNews project root. Expected either "
        f"{base_dir}\\backend or {base_dir}\\hyperpersonalisedNewsReccomendation\\backend."
    )


def resolve_backend_dir(start_file: str | Path) -> Path:
    return resolve_project_root(start_file) / "backend"


def resolve_bundle_root(start_file: str | Path) -> Path:
    project_root = resolve_project_root(start_file)
    if project_root.name == "hyperpersonalisedNewsReccomendation":
        return project_root.parent
    return project_root
