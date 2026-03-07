#!/usr/bin/env python3
"""Clean up orphaned Lakebase Autoscaling resources from interrupted deploys."""
import sys
import time

from databricks.sdk import WorkspaceClient


MAX_RETRIES = 8
INITIAL_BACKOFF_S = 5


def delete_resource(label: str, delete_fn, resource_path: str) -> bool:
    """Delete a Postgres resource and wait for the operation to complete."""
    print(f"  Deleting {label}: {resource_path}...")
    try:
        operation = delete_fn(name=resource_path)
        operation.wait()
        print(f"    Deleted.")
        return True
    except Exception as e:
        msg = str(e)
        if "NOT_FOUND" in msg or "not found" in msg.lower():
            print(f"    Not found (already deleted).")
            return True
        raise


def delete_resource_with_retry(label: str, delete_fn, resource_path: str) -> bool:
    """Delete a resource with exponential backoff for transient states.

    Retries on 'reconciliation in progress' errors — these are temporary and
    resolve once the endpoint finishes its current operation.
    """
    backoff = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return delete_resource(label, delete_fn, resource_path)
        except Exception as e:
            msg = str(e)
            is_transient = "reconciliation" in msg.lower() or "retry" in msg.lower()
            if is_transient and attempt < MAX_RETRIES:
                print(f"    Attempt {attempt}/{MAX_RETRIES}: transient error, retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                print(f"    Failed: {e}")
                return False
    return False


def list_branches(w: WorkspaceClient, project_path: str) -> list[str]:
    """List branches for a project (diagnostic — shows auto-created branches)."""
    try:
        branches = list(w.postgres.list_branches(parent=project_path))
        if branches:
            print(f"  Branches found for {project_path}:")
            for b in branches:
                print(f"    - {b.name}")
        else:
            print(f"  No branches found for {project_path}.")
        return [b.name for b in branches]
    except Exception as e:
        if "NOT_FOUND" in str(e) or "not found" in str(e).lower():
            print(f"  Project not found (already deleted) — skipping branch listing.")
        else:
            print(f"  Could not list branches: {e}")
        return []


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <profile> <project_id>")
        sys.exit(1)

    profile, project_id = sys.argv[1], sys.argv[2]
    w = WorkspaceClient(profile=profile)

    project_path = f"projects/{project_id}"

    # Delete autoscaling project (cascades to branches + endpoints)
    #
    # Individual endpoint/branch deletion does NOT work:
    #   - Read-write endpoints cannot be deleted individually
    #   - Root branches cannot be deleted individually
    # Project deletion cascades and removes everything.
    #
    # If endpoint reconciliation is in progress, we retry with backoff —
    # reconciliation is a temporary state that resolves on its own.
    print("Autoscaling resources:")

    # Diagnostic: show what exists
    list_branches(w, project_path)

    delete_resource_with_retry(
        "Autoscaling project",
        w.postgres.delete_project,
        project_path,
    )


if __name__ == "__main__":
    main()
