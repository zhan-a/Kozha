#!/usr/bin/env bash
# scripts/restore.sh — restore chat2hamnosys persistent state from a
# backup directory created by scripts/backup.sh.
#
# Usage:
#   scripts/restore.sh <src_dir>            # interactive — prompts before clobbering
#   scripts/restore.sh --verify <src_dir>   # check checksums; do not write
#   scripts/restore.sh --force <src_dir>    # non-interactive (cron / CI only)
#
# Verification step (always runs first): every file in manifest.json
# must hash-match what's on disk. Restore aborts if anything diverges.
#
# Post-restore smoke: starts uvicorn briefly against the restored DBs,
# polls /api/chat2hamnosys/health, then shuts down. Skip with --no-smoke.

set -euo pipefail

usage() {
    sed -n '2,16p' "$0" >&2
    exit 1
}

MODE="interactive"
RUN_SMOKE=1
SRC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify)   MODE="verify"; shift ;;
        --force)    MODE="force"; shift ;;
        --no-smoke) RUN_SMOKE=0; shift ;;
        -h|--help)  usage ;;
        -*)         echo "unknown flag: $1" >&2; usage ;;
        *)          SRC="$1"; shift ;;
    esac
done

[[ -n "$SRC" && -d "$SRC" ]] || usage

log() { echo "[$(date -u +%FT%TZ)] $*"; }

DATA_DIR="${CHAT2HAMNOSYS_DATA_DIR:-/app/data}"
SIGN_DB="${CHAT2HAMNOSYS_SIGN_DB:-${DATA_DIR}/authored_signs.sqlite3}"
SESSION_DB="${CHAT2HAMNOSYS_SESSION_DB:-${DATA_DIR}/chat2hamnosys/sessions.sqlite3}"
TOKEN_DB="${CHAT2HAMNOSYS_TOKEN_DB:-${DATA_DIR}/chat2hamnosys/session_tokens.sqlite3}"
REVIEWER_DB="${CHAT2HAMNOSYS_REVIEWER_DB:-${DATA_DIR}/chat2hamnosys/reviewers.sqlite3}"
AUDIT_LOG="${CHAT2HAMNOSYS_EXPORT_AUDIT:-${DATA_DIR}/chat2hamnosys/export_audit.jsonl}"

# ---------- 1. Verify manifest checksums ----------------------------------
MANIFEST="${SRC}/manifest.json"
[[ -f "$MANIFEST" ]] || { log "ERROR: manifest.json missing in ${SRC}"; exit 2; }

log "verifying manifest checksums"
python3 - "$SRC" <<'PY'
import json, hashlib, sys, pathlib
src = pathlib.Path(sys.argv[1])
manifest = json.loads((src / "manifest.json").read_text())
bad = []
for name, meta in manifest["files"].items():
    p = src / name
    if not p.exists():
        bad.append(f"{name}: missing on disk")
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != meta["sha256"]:
        bad.append(f"{name}: sha256 mismatch (manifest={meta['sha256']} disk={h})")
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
print(f"verified {len(manifest['files'])} files")
PY

if [[ "$MODE" == "verify" ]]; then
    log "verify-only mode — done"
    exit 0
fi

# ---------- 2. Prompt unless --force --------------------------------------
if [[ "$MODE" == "interactive" ]]; then
    echo
    echo "This will OVERWRITE the live state at:"
    echo "  ${SIGN_DB}"
    echo "  ${SESSION_DB}"
    echo "  ${TOKEN_DB}"
    echo "  ${REVIEWER_DB}"
    echo "  ${AUDIT_LOG}"
    echo
    read -r -p "Type 'restore' to proceed: " confirm
    [[ "$confirm" == "restore" ]] || { log "aborted"; exit 1; }
fi

# ---------- 3. Stop the service before touching the DBs -------------------
SERVICE_STOPPED=0
if systemctl is-active --quiet kozha.service 2>/dev/null; then
    log "stopping kozha.service"
    sudo systemctl stop kozha.service
    SERVICE_STOPPED=1
fi

# ---------- 4. Snapshot current state (defense-in-depth) ------------------
SAFETY="${DATA_DIR}/.restore_snapshot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAFETY"
for f in "$SIGN_DB" "$SESSION_DB" "$TOKEN_DB" "$REVIEWER_DB" "$AUDIT_LOG"; do
    [[ -f "$f" ]] && cp -p "$f" "${SAFETY}/$(basename "$f")"
done
log "current state snapshotted -> ${SAFETY} (delete after verifying restore)"

# ---------- 5. Copy restored files into place -----------------------------
restore_one() {
    local src="$1" dst="$2"
    if [[ ! -f "$src" ]]; then
        log "skip: ${src} not in backup"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst"
    log "restored -> ${dst}"
}

restore_one "${SRC}/signs.sqlite3"        "$SIGN_DB"
restore_one "${SRC}/sessions.sqlite3"     "$SESSION_DB"
restore_one "${SRC}/tokens.sqlite3"       "$TOKEN_DB"
restore_one "${SRC}/reviewers.sqlite3"    "$REVIEWER_DB"
restore_one "${SRC}/export_audit.jsonl"   "$AUDIT_LOG"

if [[ -f "${SRC}/authored_signs.tar.gz" ]]; then
    log "restoring authored_signs/"
    tar -xzf "${SRC}/authored_signs.tar.gz" -C "$DATA_DIR"
fi

# ---------- 6. Restart the service ----------------------------------------
if [[ "$SERVICE_STOPPED" -eq 1 ]]; then
    log "starting kozha.service"
    sudo systemctl start kozha.service
fi

# ---------- 7. Smoke test -------------------------------------------------
if [[ "$RUN_SMOKE" -eq 1 ]]; then
    log "smoke check: GET /api/chat2hamnosys/health"
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if curl --fail --silent http://127.0.0.1:8000/api/chat2hamnosys/health >/dev/null; then
            log "smoke check passed"
            exit 0
        fi
        sleep 3
    done
    log "ERROR: smoke check did not see a healthy /health within 30s"
    exit 4
fi

log "restore complete (skipped smoke check)"
