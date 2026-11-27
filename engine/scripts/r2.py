"""Copies a run directory to and from R2, for runs that survive the pod.

    r2.py push DIR PREFIX      upload new or changed files (size or mtime), skipping *.tmp
    r2.py get KEY DEST         download one object
    r2.py pull PREFIX DIR      download every file under PREFIX that is missing or differs in size (needs a key)

On a remote-pod pod there is no store key: $REMOTE_POD_GRANT_URL (the agent, loopback only) hands out a presigned URL
per object, which the CLI on the owner's machine signs if the pod's policy allows it (uploads under the pod's own
prefixes such as runs/<run>/, downloads of its own inputs; never a listing). With REMOTE_POD_S3_* set instead (a
machine you trust), boto3 is used with that key.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GRANTS = os.environ.get("REMOTE_POD_GRANT_URL")
KEYED = bool(os.environ.get("REMOTE_POD_S3_ACCESS_KEY"))


def _grant(op: str, key: str, size: int = 0) -> str:
    """A presigned URL from the agent (an upload's is signed for exactly `size` bytes); retried while no signer is
    connected (HTTP 503)."""
    req_body = {"op": op, "key": key}
    if size:
        req_body["size"] = size
    body = json.dumps(req_body).encode()
    for attempt in range(60):
        req = urllib.request.Request(GRANTS + "/v1/grant", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["url"]
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 503:  # no signer yet: wait for remote-pod signer / logs -f
                if attempt == 0:
                    print(f"waiting for a signer: {msg.strip()}", file=sys.stderr, flush=True)
                time.sleep(10)
                continue
            raise SystemExit(f"{op} {key}: refused: {msg.strip()}")
    raise SystemExit(f"{op} {key}: no signer for 10 minutes")


def _put(path: Path, key: str):
    if KEYED:
        _s3().upload_file(str(path), _bucket(), key)
        return
    size = path.stat().st_size
    url = _grant("put", key, size)
    with open(path, "rb") as f:
        req = urllib.request.Request(url, data=f.read(size), method="PUT", headers={"Content-Length": str(size)})
        with urllib.request.urlopen(req, timeout=3600) as r:
            r.read()


def _get(key: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if KEYED:
        _s3().download_file(_bucket(), key, str(dest))
        return
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(_grant("get", key), timeout=3600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.environ["REMOTE_POD_S3_ENDPOINT"],
                        aws_access_key_id=os.environ["REMOTE_POD_S3_ACCESS_KEY"],
                        aws_secret_access_key=os.environ["REMOTE_POD_S3_SECRET_KEY"],
                        region_name=os.environ.get("REMOTE_POD_S3_REGION") or "auto")


def _bucket() -> str:
    return os.environ["REMOTE_POD_S3_BUCKET"]


def push(root: Path, prefix: str):
    state_file = root / ".r2-pushed.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.endswith((".tmp", ".part")) or p == state_file:
            continue
        rel = p.relative_to(root).as_posix()
        st = p.stat()
        sig = [st.st_size, int(st.st_mtime)]
        if state.get(rel) == sig:
            continue
        _put(p, f"{prefix}/{rel}")
        state[rel] = sig
        n += 1
    state_file.write_text(json.dumps(state))
    print(f"pushed {n} file(s) to {prefix}", flush=True)


def pull(prefix: str, root: Path):
    if not KEYED:
        raise SystemExit("pull lists the store, which a pod may not do: r2.py get KEY DEST for each file")
    s3, n = _s3(), 0
    base = root.resolve()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=_bucket(), Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            local = (root / o["Key"][len(prefix) + 1:]).resolve()
            if base not in local.parents:  # an object named "../x" would land outside root
                print(f"skipped {o['Key']!r}: outside {root}", file=sys.stderr)
                continue
            if local.exists() and local.stat().st_size == o["Size"]:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(_bucket(), o["Key"], str(local))
            n += 1
    print(f"pulled {n} file(s) from {prefix}", flush=True)


if __name__ == "__main__":
    if not KEYED and not GRANTS:
        raise SystemExit("no store access: neither REMOTE_POD_GRANT_URL (a remote-pod pod) nor REMOTE_POD_S3_* is set")
    cmd, a, b = sys.argv[1:4]
    if cmd == "push":
        push(Path(a), b)
    elif cmd == "get":
        _get(a, Path(b))
    elif cmd == "pull":
        pull(a, Path(b))
    else:
        raise SystemExit(__doc__)
