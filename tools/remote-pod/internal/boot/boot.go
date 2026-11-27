// Package boot is how a pod on a public image starts the agent: the bootstrap script (run by the image's command on
// RunPod, by the instance's onstart on vast.ai) downloads the agent, and lbc when given, through presigned URLs, checks
// each against the sha256 fixed when the pod was created, and runs the agent. Nothing here holds a store key.
package boot

// Script fetches the binaries with whatever the image has (curl, wget, python3, else apt installs curl), refuses
// any whose sha256 differs from REMOTE_POD_AGENT_SHA256 / REMOTE_POD_LBC_SHA256 (deleting it), and replaces itself
// with the agent. REMOTE_POD_BIN_DIR (default /usr/local/bin) is where they go. A restart runs it again, so the URLs
// must still be valid then (7 days).
const Script = `set -e
fetch() {
  if command -v curl >/dev/null; then curl -fsSL "$1" -o "$2"
  elif command -v wget >/dev/null; then wget -qO "$2" "$1"
  elif command -v python3 >/dev/null; then python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
  else apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates >/dev/null && curl -fsSL "$1" -o "$2"
  fi
}
# each binary must match the sha256 fixed when the pod was created (the store's copy could have been replaced)
check() {
  if command -v sha256sum >/dev/null; then echo "$2  $1" | sha256sum -c --quiet - || { echo "sha256 mismatch: $1" >&2; rm -f "$1"; exit 1; }
  else python3 -c 'import hashlib, sys; sys.exit(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != sys.argv[2])' "$1" "$2" || { echo "sha256 mismatch: $1" >&2; rm -f "$1"; exit 1; }
  fi
}
BIN="${REMOTE_POD_BIN_DIR:-/usr/local/bin}"
mkdir -p "$BIN"
fetch "$REMOTE_POD_AGENT_URL" "$BIN/remote-pod"
check "$BIN/remote-pod" "$REMOTE_POD_AGENT_SHA256"
chmod +x "$BIN/remote-pod"
if [ -n "${REMOTE_POD_LBC_URL:-}" ]; then
  fetch "$REMOTE_POD_LBC_URL" "$BIN/lbc"
  check "$BIN/lbc" "$REMOTE_POD_LBC_SHA256"
  chmod +x "$BIN/lbc"
fi
mkdir -p "$REMOTE_POD_WORKDIR"
cd "$REMOTE_POD_WORKDIR"
exec "$BIN/remote-pod" agent
`

// VastOnstart starts the bootstrap on vast.ai, whose own init (SSH) runs as the container's command: the instance's
// onstart script runs it in the background with the container's whole environment restored from /proc/1/environ (the
// image's PATH and CUDA variables and the ones given at creation live there, whatever shell runs the script).
const VastOnstart = `nohup bash -c 'while IFS= read -r -d "" kv; do export "$kv"; done </proc/1/environ; export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"; echo "$REMOTE_POD_BOOTSTRAP" | base64 -d | bash' >/var/log/remote-pod.log 2>&1 &`
