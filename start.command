#!/bin/zsh
cd -- "${0:A:h}"
python3 backend/server.py --port 4173
