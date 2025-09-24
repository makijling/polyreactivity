"""Utilities to push the Space assets to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


SPACE_FILES = {
    "space/app.py": "app.py",
    "space/README.md": "README.md",
    "space/runtime.yaml": "runtime.yaml",
    "requirements.txt": "requirements.txt",
    "pyproject.toml": "pyproject.toml",
    "LICENSE": "LICENSE",
    "configs/default.yaml": "configs/default.yaml",
    "space/artifacts/model.joblib": "artifacts/model.joblib",
}

SPACE_FOLDERS = {
    "polyreact": "polyreact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy polyreact Space to Hugging Face")
    parser.add_argument("--space-id", required=True, help="Target Space id, e.g. username/polyreactivity")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the Space as private (default public)",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face token (default: HF_TOKEN)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv(args.token_env)
    if not token:
        raise SystemExit(
            f"Missing Hugging Face token. Set the {args.token_env} environment variable before running."
        )

    repo_id = args.space_id
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )

    repo_kwargs = {"repo_type": "space", "repo_id": repo_id, "token": token}

    for local, remote in SPACE_FILES.items():
        local_path = Path(local)
        if not local_path.exists():
            raise FileNotFoundError(f"Missing required file for Space deployment: {local}")
        api.upload_file(path_or_fileobj=local_path, path_in_repo=remote, **repo_kwargs)

    for local, remote in SPACE_FOLDERS.items():
        local_path = Path(local)
        if not local_path.exists():
            raise FileNotFoundError(f"Missing required folder for Space deployment: {local}")
        upload_folder(
            repo_id=repo_id,
            repo_type="space",
            token=token,
            folder_path=str(local_path),
            path_in_repo=remote,
        )

    print(f"Space {repo_id} updated successfully.")


if __name__ == "__main__":
    main()
