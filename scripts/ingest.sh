#!/usr/bin/env bash
set -euo pipefail
python -m app.rag.ingestion --paths data/sample_medical.md data/sample_finance.md --domain medical
