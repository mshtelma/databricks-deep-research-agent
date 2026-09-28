#!/bin/sh

set -eu

index_url=${PYPI_PROXY_URL:-${UV_DEFAULT_INDEX:-}}

if [ -n "$index_url" ]; then
    case "$index_url" in
        http://*|https://*) ;;
        *)
            echo "ERROR: Python package index must be an http(s) PEP 503 URL without credentials or whitespace." >&2
            exit 2
            ;;
    esac

    case "$index_url" in
        *[[:space:]]*)
            echo "ERROR: Python package index must be an http(s) PEP 503 URL without credentials or whitespace." >&2
            exit 2
            ;;
    esac

    authority=${index_url#*://}
    authority=${authority%%/*}
    case "$authority" in
        ""|*@*)
            echo "ERROR: Python package index must be an http(s) PEP 503 URL without credentials or whitespace." >&2
            exit 2
            ;;
    esac

    case "$index_url" in
        *\?*|*\#*)
            echo "ERROR: Python package index must not contain a query string or fragment." >&2
            exit 2
            ;;
    esac

    UV_DEFAULT_INDEX=$index_url
    export UV_DEFAULT_INDEX
fi

exec "$@"
