"""Conformance delegation — run the bench suite against this app.

    From ../taste-bench:
      TASTE_APP=taste.api:create_app TASTE_APP_FACTORY=1 \
        .venv/bin/python -m pytest conformance/

This file exists so `pytest tests/` inside the repo stays green standalone;
the authoritative run happens from taste-bench.
"""
