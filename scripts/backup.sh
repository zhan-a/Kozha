#!/usr/bin/env bash
# scripts/backup.sh — daily backup of chat2hamnosys persistent state.
#
# Usage:
#   scripts/backup.sh <dest_dir>
#
# Example (cron, daily at 02:00):
#   0 2 * * * /opt/kozha/scripts/backup.sh /backups/$(date +\%Y-\%m-\%d) >> /var/log/kozha/backup.log 2>&1
#
# What gets backed up (in <dest_dir>/):
#   - signs.sqlite3        — production sign DB (online via .backup, never cp)
#   - sessions.sqlite3
#   - tokens.sqlite3
#   - reviewers.sqlite3
#   - export_audit.jsonl   — append-only audit trail
#   - authored_signs/      — per-language SiGML files (if present)
#   - manifest.json        — sha256s, sizes, source paths, BUILD_SHA
#
# Exit codes: 0 success, 1 invalid args, 2 source missing, 3 sqlite error,
# 4 verification failure.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dest_dir>" >&2
    exit 1
fi

DEST="$1"
DATA_DIR="${CHAT2HAMNOSYS_DATA_DIR:-/app/data}"
SIGN_DB="${CHAT2HAMNOSYS_SIGN_DB:-${DATA_DIR}/authored_signs.sqlite3}"
SESSION_DB="${CHAT2HAMNOSYS_SESSION_DB:-${DATA_DIR}/chat2hamnosys/sessions.sqlite3}"
TOKEN_DB="${CHAT2HAMNOSYS_TOKEN_DB:-${DATA_DIR}/chat2hamnosys/session_tokens.sqlite3}"
REVIEWER_DB="${CHAT2HAMNOSYS_REVIEWER_DB:-${DATA_DIR}/chat2hamnosys/reviewers.sqlite3}"
AUDIT_LOG="${CHAT2HAMNOSYS_EXPORT_AUDIT:-${DATA_DIR}/chat2hamnosys/export_audit.jsonl}"
AUTHORED_DIR="${DATA_DIR}/authored_signs"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

mkdir -p "$DEST"

backup_sqlite() {
    local src="$1" name="$2"
    if [[ ! -f "$src" ]]; then
        log "skip ${name}: source missing (${src})"
        return 0
    fi
    log "backup ${name} (${src})"
    # Use sqlite's online .backup so a writer mid-transaction doesn't
    # corrupt the copy. Falls back to file copy if sqlite3 isn't on PATH.
    if command -v sqlite3 >/dev/null; then
        sqlite3 "$src" ".backup '${DEST}/${name}'" || {
            log "ERROR: sqlite3 .backup failed for ${name}"; exit 3;
        }
    else
        cp -p "$src" "${DEST}/${name}"
    fi
}

backup_sqlite "$SIGN_DB"     "signs.sqlite3"
backup_sqlite "$SESSION_DB"  "sessions.sqlite3"
backup_sqlite "$TOKEN_DB"    "tokens.sqlite3"
backup_sqlite "$REVIEWER_DB" "reviewers.sqlite3"

# Append-only JSONL audit log — copy + verify line count matches.
if [[ -f "$AUDIT_LOG" ]]; then
    log "backup export_audit.jsonl (${AUDIT_LOG})"
    cp -p "$AUDIT_LOG" "${DEST}/export_audit.jsonl"
    src_lines=$(wc -l < "$AUDIT_LOG")
    dst_lines=$(wc -l < "${DEST}/export_audit.jsonl")
    [[ "$src_lines" == "$dst_lines" ]] || { log "ERROR: audit-log line counts diverged"; exit 4; }
fi

# Authored SiGML directory — per-language XML files. tar+gzip for
# compactness; manifest records the contents.
if [[ -d "$AUTHORED_DIR" ]]; then
    log "backup authored_signs/ (${AUTHORED_DIR})"
    tar -czf "${DEST}/authored_signs.tar.gz" -C "$(dirname "$AUTHORED_DIR")" "$(basename "$AUTHORED_DIR")"
fi

# Manifest with checksums — restore script verifies these before clobbering.
log "writing manifest"
{
    printf '{\n'
    printf '  "created_at": "%s",\n' "$(date -u +%FT%TZ)"
    printf '  "build_sha": "%s",\n' "${BUILD_SHA:-unknown}"
    printf '  "host": "%s",\n' "$(hostname)"
    printf '  "files": {\n'
    first=1
    for f in "${DEST}"/*; do
        name=$(basename "$f")
        [[ "$name" == "manifest.json" ]] && continue
        if [[ "$first" -eq 1 ]]; then first=0; else printf ',\n'; fi
        sha=$(shasum -a 256 "$f" | cut -d' ' -f1)
        size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
        printf '    "%s": {"sha256": "%s", "bytes": %s}' "$name" "$sha" "$size"
    done
    printf '\n  }\n}\n'
} > "${DEST}/manifest.json"

log "backup complete -> ${DEST}"
log "verify with: scripts/restore.sh --verify ${DEST}"
