#!/bin/sh

set -eu

repo_root=$(pwd)
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
index_wrapper="$script_dir/with_pypi_index.sh"
lock_scope=${LOCK_SCOPE:-all}
uv_bin=${UV_BIN:-uv}
relock_upgrade=${RELOCK_UPGRADE:-0}
allow_default_relock=${ALLOW_DEFAULT_RELOCK:-0}
index_url=${PYPI_PROXY_URL:-${UV_DEFAULT_INDEX:-}}

case "$lock_scope" in
    all) selected="workspace framework app" ;;
    workspace|framework|app) selected=$lock_scope ;;
    *)
        echo "ERROR: LOCK_SCOPE must be one of: all, workspace, framework, app." >&2
        exit 2
        ;;
esac

case "$relock_upgrade" in
    0|no|false|off|"") upgrade_arg="" ;;
    1|yes|true|on) upgrade_arg="--upgrade" ;;
    *)
        echo "ERROR: RELOCK_UPGRADE must be a boolean value." >&2
        exit 2
        ;;
esac

case "$allow_default_relock" in
    0|no|false|off|"")
        if [ -z "$index_url" ]; then
            echo "ERROR: relock requires PYPI_PROXY_URL or UV_DEFAULT_INDEX." >&2
            exit 2
        fi
        ;;
    1|yes|true|on) ;;
    *)
        echo "ERROR: ALLOW_DEFAULT_RELOCK must be a boolean value." >&2
        exit 2
        ;;
esac

"$index_wrapper" true

lock_dir="$repo_root/.uv-relock-in-progress"
tmp_dir=""
completed=0

project_dir() {
    case "$1" in
        workspace) printf '%s\n' "$repo_root" ;;
        framework) printf '%s\n' "$repo_root/databricks-deep-research" ;;
        app) printf '%s\n' "$repo_root/databricks-deep-research-app" ;;
    esac
}

restore_locks() {
    [ -n "$tmp_dir" ] || return 0
    for project in $selected; do
        directory=$(project_dir "$project")
        backup="$tmp_dir/$project.uv.lock"
        if [ -f "$backup" ]; then
            cp "$backup" "$directory/uv.lock"
        fi
    done
}

cleanup() {
    status=$?
    if [ "$completed" -ne 1 ]; then
        restore_locks
    fi
    if [ -n "$tmp_dir" ]; then
        rm -rf "$tmp_dir"
    fi
    rm -f "$lock_dir/pid"
    rmdir "$lock_dir" 2>/dev/null || true
    exit "$status"
}

if ! mkdir "$lock_dir" 2>/dev/null; then
    if [ -f "$lock_dir/pid" ]; then
        lock_pid=$(cat "$lock_dir/pid" 2>/dev/null || true)
        case "$lock_pid" in
            *[!0-9]*|"") lock_pid="" ;;
        esac
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            echo "ERROR: another uv.lock recreation is already in progress." >&2
            exit 1
        fi
    fi
    rm -f "$lock_dir/pid"
    if ! rmdir "$lock_dir" 2>/dev/null || ! mkdir "$lock_dir" 2>/dev/null; then
        echo "ERROR: another uv.lock recreation is already in progress." >&2
        exit 1
    fi
fi
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s\n' "$$" > "$lock_dir/pid"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/deep-research-relock.XXXXXX")

for project in $selected; do
    directory=$(project_dir "$project")
    lockfile="$directory/uv.lock"
    if [ ! -s "$lockfile" ]; then
        echo "ERROR: expected lockfile is missing for $project." >&2
        exit 1
    fi
    cp "$lockfile" "$tmp_dir/$project.uv.lock"
done

for project in $selected; do
    directory=$(project_dir "$project")
    echo "Recreating $project uv.lock..."
    if [ -n "$upgrade_arg" ]; then
        (cd "$directory" && "$index_wrapper" "$uv_bin" lock "$upgrade_arg")
    else
        (cd "$directory" && "$index_wrapper" "$uv_bin" lock)
    fi
    if [ ! -s "$directory/uv.lock" ] || ! grep -Eq '^version = [0-9]+' "$directory/uv.lock"; then
        echo "ERROR: uv produced an invalid lockfile for $project." >&2
        exit 1
    fi
done

for project in $selected; do
    directory=$(project_dir "$project")
    echo "Checking $project uv.lock..."
    (cd "$directory" && "$index_wrapper" "$uv_bin" lock --check)
    if grep -Eq 'https?://[^/[:space:]]+@|[?&](token|access_token|password|secret)=' "$directory/uv.lock"; then
        echo "ERROR: credential-like data found in $project uv.lock." >&2
        exit 1
    fi
done

completed=1
echo "SUCCESS: recreated and checked selected uv.lock files."
