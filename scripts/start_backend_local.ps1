param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:DATABASE_URL = "sqlite:///./data/hypernews.db"
$env:HYPERNEWS_ENABLE_SENTENCE_TRANSFORMER = "0"
$env:HYPERNEWS_ENABLE_RERANKER = "0"

Set-Location (Join-Path $PSScriptRoot "..")
python -m uvicorn backend.app:app --host 127.0.0.1 --port $Port
