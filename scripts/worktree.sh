#!/usr/bin/env bash
# =============================================================================
# Git Worktree Manager
# =============================================================================
# Creates isolated worktrees with shared .env files for parallel development.
# All gitignored .env* files are auto-symlinked from the main worktree.
#
# Usage:
#   worktree.sh create <branch> [base] [--install]
#   worktree.sh list
#   worktree.sh remove <branch> [--delete-branch]
#   worktree.sh link-env <branch>
# =============================================================================
set -euo pipefail

WORKTREE_DIR_NAME=".worktrees"

die() { echo "Error: $*" >&2; exit 1; }

# Always resolves to the MAIN worktree, even when called from a worktree.
get_main_worktree() {
    git rev-parse --show-toplevel &>/dev/null || die "not inside a git repository"
    # Avoid SIGPIPE: capture full output first, then extract first line.
    # (head -1 in a pipeline under pipefail can cause SIGPIPE on the upstream command.)
    local output
    output=$(git worktree list --porcelain)
    echo "$output" | sed -n '1s/^worktree //p'
}

get_worktree_container() {
    local main="$1"
    echo "$(dirname "$main")/$WORKTREE_DIR_NAME"
}

# Replace / with - for safe directory names.
sanitize_branch() {
    echo "$1" | tr '/' '-'
}

# Symlink all untracked .env* files from main → target worktree.
# Skips committed .env files (already in worktree via git).
# Never creates broken symlinks.
link_env_files() {
    local main="$1" target_dir="$2"
    local linked=0

    for src in "$main"/.env*; do
        [ -e "$src" ] || continue
        local name
        name=$(basename "$src")
        [ "$name" = ".env.example" ] && continue

        # Only symlink files NOT tracked by git (committed files are already in worktree).
        # This catches both gitignored files (.env) and untracked-but-not-ignored files
        # (.env.local-dev etc. that have gitignore exceptions but aren't committed).
        if ! git -C "$main" ls-files --error-unmatch "$name" &>/dev/null; then
            ln -sf "$src" "$target_dir/$name"
            echo "  Linked $name"
            linked=$((linked + 1))
        fi
    done

    if [ "$linked" -eq 0 ]; then
        echo "  No untracked .env files found to link."
        echo "  Create .env first, then run: make worktree-link BRANCH=<name>"
    fi
}

# --- Subcommands ---

cmd_create() {
    local branch="${1:?Usage: worktree.sh create <branch> [base-ref] [--install]}"
    shift
    local base=""
    local do_install=false

    while [ $# -gt 0 ]; do
        case "$1" in
            --install) do_install=true ;;
            *)         base="$1" ;;
        esac
        shift
    done

    # Default base = current branch
    if [ -z "$base" ]; then
        base=$(git rev-parse --abbrev-ref HEAD)
    fi

    local main dir_name container worktree_path
    main=$(get_main_worktree)
    dir_name=$(sanitize_branch "$branch")
    container=$(get_worktree_container "$main")
    worktree_path="$container/$dir_name"

    [ -d "$worktree_path" ] && die "directory already exists: $worktree_path
  Use 'make worktree-list' to see worktrees.
  Use 'make worktree-remove BRANCH=$branch' to remove it first."

    mkdir -p "$container"

    # Detect branch state and handle "already checked out" case
    if git rev-parse --verify "$branch" &>/dev/null; then
        local current_branch
        current_branch=$(git -C "$main" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
        if [ "$current_branch" = "$branch" ]; then
            die "branch '$branch' is currently checked out in the main worktree.
  Switch to a different branch first:  git checkout main
  Then retry:                          make worktree BRANCH=$branch"
        fi
        echo "Checking out existing branch '$branch'..."
        git worktree add "$worktree_path" "$branch"
    elif git rev-parse --verify "origin/$branch" &>/dev/null; then
        echo "Checking out remote branch 'origin/$branch'..."
        git worktree add "$worktree_path" "$branch"
    else
        echo "Creating new branch '$branch' from '$base'..."
        git worktree add -b "$branch" "$worktree_path" "$base"
    fi

    # Symlink all untracked .env* files
    echo "Linking environment files..."
    link_env_files "$main" "$worktree_path"

    # Optional: install dependencies
    if [ "$do_install" = true ]; then
        echo "Installing dependencies..."
        ( cd "$worktree_path" && make install )
    fi

    echo ""
    echo "Worktree ready: $worktree_path"
    echo ""
    echo "  cd $worktree_path"
    if [ "$do_install" = false ]; then
        echo "  make install        # required before first use (or re-create with INSTALL=1)"
    fi
    echo "  make dev            # start dev server"
    echo "  make test           # run tests"
    echo ""
    echo "  Different port:  PORT=8001 make dev"
}

cmd_list() {
    git worktree list
}

cmd_remove() {
    local branch="${1:?Usage: worktree.sh remove <branch> [--delete-branch]}"
    local delete_branch=false
    if [ "${2:-}" = "--delete-branch" ]; then
        delete_branch=true
    fi

    local main dir_name container worktree_path
    main=$(get_main_worktree)
    dir_name=$(sanitize_branch "$branch")
    container=$(get_worktree_container "$main")
    worktree_path="$container/$dir_name"

    [ -d "$worktree_path" ] || die "no worktree at $worktree_path
$(git worktree list)"

    echo "Removing worktree at $worktree_path..."
    # Use --force because our symlinked .env files are always seen as untracked.
    # Git still protects against removing worktrees with uncommitted changes to
    # tracked files (would need a second --force for that).
    if ! git worktree remove --force "$worktree_path" 2>&1; then
        echo ""
        echo "Worktree has uncommitted changes to tracked files. Options:"
        echo "  1. Commit or stash changes first"
        echo "  2. Force: git worktree remove --force --force $worktree_path"
        exit 1
    fi

    if [ "$delete_branch" = true ]; then
        if ! git branch -d "$branch" 2>/dev/null; then
            echo "Warning: branch '$branch' not deleted (not fully merged). Use 'git branch -D $branch' to force."
        else
            echo "Deleted branch '$branch'."
        fi
    fi

    echo "Done."
}

cmd_link_env() {
    local branch="${1:?Usage: worktree.sh link-env <branch>}"

    local main dir_name container worktree_path
    main=$(get_main_worktree)
    dir_name=$(sanitize_branch "$branch")
    container=$(get_worktree_container "$main")
    worktree_path="$container/$dir_name"

    [ -d "$worktree_path" ] || die "no worktree at $worktree_path"

    link_env_files "$main" "$worktree_path"
    echo "Done."
}

# --- Dispatch ---
case "${1:-help}" in
    create)    shift; cmd_create "$@" ;;
    list)      cmd_list ;;
    remove)    shift; cmd_remove "$@" ;;
    link-env)  shift; cmd_link_env "$@" ;;
    help)
        echo "Usage: worktree.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  create <branch> [base] [--install]   Create worktree from base (default: current branch)"
        echo "  list                                  List all worktrees"
        echo "  remove <branch> [--delete-branch]     Remove worktree"
        echo "  link-env <branch>                     Re-link env files"
        exit 0
        ;;
    *)
        echo "Unknown command: $1" >&2
        echo "Run 'worktree.sh help' for usage." >&2
        exit 1
        ;;
esac
