param(
    [int]$Port = 8000,
    [int]$MaxArticles = 0,
    [switch]$DisableGraph,
    [switch]$UseDenseModels
)

$env:PYTHONUTF8 = "1"
$env:HYPERNEWS_MAX_ARTICLES = "$MaxArticles"
$env:HYPERNEWS_ENABLE_GRAPH = $(if ($DisableGraph) { "0" } else { "1" })
$env:HYPERNEWS_ENABLE_SENTENCE_TRANSFORMER = $(if ($UseDenseModels) { "1" } else { "0" })
$env:HYPERNEWS_ENABLE_RERANKER = $(if ($UseDenseModels) { "1" } else { "0" })

python -m uvicorn backend.app:app --host 127.0.0.1 --port $Port
