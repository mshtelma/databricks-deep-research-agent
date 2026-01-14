"""Complex, long-running research tests.

These tests use production configuration (config/app.yaml) and run
full research flows with multiple iterations. They are opt-in and
should only be run manually or in nightly CI builds.

Run with:
    make test-complex
    uv run pytest tests/complex -v -s --timeout=600
"""
