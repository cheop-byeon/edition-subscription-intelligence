#!/bin/zsh
cd -- "${0:A:h}"
if [[ -n "$EDITION_PYTHON" ]]; then
  exec "$EDITION_PYTHON" scripts/run_edition.py
elif [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python scripts/run_edition.py
elif [[ -x .venv-analytics/bin/python ]]; then
  exec .venv-analytics/bin/python scripts/run_edition.py
else
  python3 scripts/run_edition.py
fi
