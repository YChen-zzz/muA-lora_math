"""Download a Hugging Face repository with retries.

Usage: python test3.py MODEL_OR_DATASET_ID [--repo-type model|dataset]
Credentials, when needed, come from the standard Hugging Face login or HF_TOKEN.
"""

import argparse
import time

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--local-dir")
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    for attempt in range(1, args.max_retries + 1):
        try:
            path = snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                local_dir=args.local_dir,
                max_workers=4,
            )
            print(f"Downloaded {args.repo_type} {args.repo_id} to {path}", flush=True)
            return
        except Exception as exc:
            print(f"Attempt {attempt}/{args.max_retries} failed: {exc}", flush=True)
            if attempt == args.max_retries:
                raise
            time.sleep(min(2**attempt, 30))


if __name__ == "__main__":
    main()
