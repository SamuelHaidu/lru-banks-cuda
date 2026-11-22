#!/usr/bin/env python3
"""Uploads code + one dataset + its tokenizer to a Runpod network volume through its S3-compatible API,
with parallel workers, so no Pod has to run (and bill) during the upload.

Usage (credentials from the environment or Runpod's downloaded credentials file; never uploaded):

    AWS_SHARED_CREDENTIALS_FILE=credentials.txt AWS_PROFILE=runpods3 \\
    uv run --with boto3 python s3_push.py --volume <network-volume-id> --dc US-KS-2 -d tinystories-10k
    (or --code-only instead of -d: just the code)

Lands in s3://<volume>/rnn-research/ = /workspace/rnn-research on a Pod. On the Pod, copy it to the local disk
before training (`cp -r /workspace/rnn-research /root/`): random reads of the memory-mapped dataset over the
network volume are slow.

Same filters as push.sh: never env files (.env, .env.*, *.env), virtual environments, runs/, .git, caches.
Files already on the volume with the same size are skipped (--force uploads everything).
"""

import argparse
import fnmatch
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".venv", "venv", "env", "runs", "datasets", "tokenizers", ".sycl_cache", "__pycache__", ".git",
             "node_modules"}
SKIP_DIR_PATTERNS = (".venv-*", "*.egg-info")
SKIP_FILES = (".env", ".env.*", "*.env", "credentials*", "*.pem", "*.key")  # secrets
SKIP_PATHS = ("studies/*.pdf",)


def code_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = Path(dirpath).relative_to(ROOT)
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not any(fnmatch.fnmatch(d, p) for p in SKIP_DIR_PATTERNS)]
        for name in filenames:
            rel = (rel_dir / name).as_posix()
            if any(fnmatch.fnmatch(name, p) for p in SKIP_FILES) or any(fnmatch.fnmatch(rel, p) for p in SKIP_PATHS):
                continue
            yield ROOT / rel_dir / name


def tree_files(top: Path):
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            yield Path(dirpath) / name


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volume", required=True, help="Network volume ID (the S3 bucket).")
    p.add_argument("--dc", required=True, help="Datacenter ID, e.g. US-KS-2 (endpoint s3api-<dc>.runpod.io).")
    p.add_argument("-d", "--dataset", help="Dataset under datasets/ (its tokenizer goes too).")
    p.add_argument("--code-only", action="store_true", help="Upload only the code, no dataset or tokenizer.")
    p.add_argument("--prefix", default="rnn-research", help="Folder on the volume (default rnn-research).")
    p.add_argument("--workers", type=int, default=16, help="Files uploaded in parallel.")
    p.add_argument("--part-mb", type=int, default=64, help="Multipart part size (Runpod max 500).")
    p.add_argument("--part-workers", type=int, default=8, help="Parallel parts per large file.")
    p.add_argument("--force", action="store_true", help="Upload even if the same size is already there.")
    p.add_argument("--dry-run", action="store_true", help="List what would be uploaded, no network.")
    args = p.parse_args()

    if not args.code_only and not args.dataset:
        p.error("-d/--dataset is required unless --code-only")
    files = list(code_files())
    tops = () if args.code_only else (ROOT / "datasets" / args.dataset, ROOT / "tokenizers" / args.dataset)
    for top in tops:
        if not top.is_dir():
            print(f"Missing {top.relative_to(ROOT)}", file=sys.stderr)
            return 1
        files += list(tree_files(top))
    jobs = [(f, f"{args.prefix}/{f.relative_to(ROOT).as_posix()}", f.stat().st_size) for f in files]
    total = sum(s for _, _, s in jobs)
    print(f"{len(jobs)} files, {total / 1e6:.1f} MB -> s3://{args.volume}/{args.prefix}/ ({args.dc})")
    if args.dry_run:
        for _, key, size in sorted(jobs, key=lambda j: -j[2])[:15]:
            print(f"  {size / 1e6:9.2f} MB  {key}")
        return 0

    import boto3

    if boto3.Session().get_credentials() is None:
        print("No credentials: set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or AWS_SHARED_CREDENTIALS_FILE +"
              " AWS_PROFILE (e.g. Runpod's downloaded credentials file, profile runpods3).", file=sys.stderr)
        return 1

    from botocore.config import Config
    from botocore.exceptions import ClientError

    s3 = boto3.client(
        "s3", region_name=args.dc, endpoint_url=f"https://s3api-{args.dc.lower()}.runpod.io/",
        config=Config(retries={"max_attempts": 10, "mode": "standard"}, read_timeout=7200,
                      max_pool_connections=args.workers * args.part_workers + 4))
    part = args.part_mb * 2**20

    sent, lock, start, finished = [0], threading.Lock(), time.time(), threading.Event()

    def add(n):  # bytes sent (or skipped), for the progress line
        with lock:
            sent[0] += n

    def progress():
        while not finished.wait(2):
            el = time.time() - start
            rate = sent[0] / max(el, 1e-9)
            eta = (total - sent[0]) / rate if rate > 0 else float("inf")
            print(f"  progress {sent[0] / 1e6:7.1f}/{total / 1e6:.1f} MB  {100 * sent[0] / max(total, 1):5.1f}%"
                  f"  {rate / 1e6:5.2f} MB/s  eta {eta:5.0f} s", flush=True)

    def retried(what, fn):  # botocore does not retry dropped TLS connections (e.g. bad record mac)
        for attempt in range(1, 11):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if attempt == 10:
                    raise
                print(f"  retry {attempt}/9 {what}: {str(exc)[-90:]}", file=sys.stderr, flush=True)
                time.sleep(min(2 * attempt, 20))

    def upload(path, key, size):
        if not args.force:
            try:
                if s3.head_object(Bucket=args.volume, Key=key)["ContentLength"] == size:
                    add(size)
                    return "skip", key, size
            except ClientError:
                pass
        if size <= part:
            retried(key, lambda: s3.upload_file(str(path), args.volume, key))
            add(size)
            return "up", key, size
        # Own multipart upload: a dropped connection retries that part only, not the whole file.
        uid = s3.create_multipart_upload(Bucket=args.volume, Key=key)["UploadId"]

        def put(n):
            with open(path, "rb") as f:
                f.seek((n - 1) * part)
                body = f.read(part)
            etag = retried(f"{key} part {n}", lambda: s3.upload_part(
                Bucket=args.volume, Key=key, UploadId=uid, PartNumber=n, Body=body)["ETag"])
            add(len(body))
            return {"PartNumber": n, "ETag": etag}

        try:
            with ThreadPoolExecutor(max_workers=args.part_workers) as parts:
                done = list(parts.map(put, range(1, -(-size // part) + 1)))
            s3.complete_multipart_upload(Bucket=args.volume, Key=key, UploadId=uid, MultipartUpload={"Parts": done})
        except Exception:
            s3.abort_multipart_upload(Bucket=args.volume, Key=key, UploadId=uid)
            raise
        return "up", key, size

    threading.Thread(target=progress, daemon=True).start()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload, *j) for j in sorted(jobs, key=lambda j: -j[2])]  # big files first
        for fut in as_completed(futures):
            try:
                status, key, size = fut.result()
            except Exception as exc:  # noqa: BLE001 - report every failure, keep the rest going
                failed += 1
                print(f"FAILED {exc}", file=sys.stderr, flush=True)
                continue
            if status == "up" or size > 1e6:
                print(f"{status:4s} {size / 1e6:8.2f} MB  {key}", flush=True)
    finished.set()
    print(f"done in {time.time() - start:.0f} s, {sent[0] / 1e6:.1f} MB, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
