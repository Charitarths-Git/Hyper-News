# Local Graph Cache

This folder is used for generated knowledge graph cache files such as
`knowledge_graph.pkl`. These cache artifacts are kept out of GitHub.

Build the cache locally with:

```powershell
python build_knowledge_graph.py
```

The graph build depends on `data/articles.parquet`, so generate the data assets
first if they are missing.
