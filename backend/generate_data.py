"""
generate_data.py
Builds the article parquet, embedding matrix, and FAISS index.

If raw MIND files are present under data/mind_full/MIND-small or data/mind,
they are preferred so entity metadata is preserved. Otherwise the script falls
back to a small synthetic demo corpus.
"""

from __future__ import annotations

import argparse
import os
import sys

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from backend.mind_data import load_mind_news, resolve_default_mind_paths

DATA_DIR = os.path.join(_BASE, "data")
PARQUET = os.path.join(DATA_DIR, "articles.parquet")
LEGACY_PARQUET = os.path.join(DATA_DIR, "news_processed.parquet")
EMB_FILE = os.path.join(DATA_DIR, "article_embeddings.npy")
FAISS_FILE = os.path.join(DATA_DIR, "faiss_mind.index")
LEGACY_FAISS_FILE = os.path.join(DATA_DIR, "news_faiss.index")

ARTICLES = [
    {"id": "T001", "category": "technology", "title": "OpenAI Releases GPT-5 with 1 Trillion Parameters", "abstract": "OpenAI has unveiled GPT-5, its most powerful language model yet, boasting 1 trillion parameters and unprecedented reasoning capabilities that outperform human experts in most benchmarks."},
    {"id": "T002", "category": "technology", "title": "Apple Vision Pro 2 Arrives with Holographic Display", "abstract": "Apple's second-generation Vision Pro headset introduces true holographic projection, spatial computing apps, and a lighter form factor aimed at mass-market adoption."},
    {"id": "T003", "category": "technology", "title": "Google DeepMind Achieves AGI Milestone in Protein Research", "abstract": "DeepMind researchers announced a major breakthrough after their AI system autonomously designed novel proteins that defeated antibiotic-resistant bacteria in lab trials."},
    {"id": "T004", "category": "technology", "title": "Meta Launches Llama 4 as Open-Source Foundation Model", "abstract": "Meta released Llama 4 with a permissive license, enabling developers worldwide to build custom AI applications without expensive API calls or proprietary restrictions."},
    {"id": "S001", "category": "sports", "title": "India Wins ICC Cricket World Cup in a Thrilling Final", "abstract": "India claimed the ICC Cricket World Cup defeating Australia by 6 runs in a nail-biting final at Mumbai's Wankhede Stadium, with Virat Kohli scoring a historic century."},
    {"id": "S002", "category": "sports", "title": "Messi Announces Retirement After Record 10th Ballon d'Or", "abstract": "Lionel Messi announced his retirement from professional football after receiving an unprecedented tenth Ballon d'Or award, capping the greatest career in the sport's history."},
    {"id": "P001", "category": "politics", "title": "UN Security Council Passes Historic Climate Emergency Resolution", "abstract": "All 15 UN Security Council members unanimously passed a binding climate emergency resolution, requiring nations to achieve net-zero emissions by 2035 or face economic sanctions."},
    {"id": "P002", "category": "politics", "title": "US Congress Passes Comprehensive AI Regulation Act", "abstract": "The US Congress passed the landmark AI Safety and Accountability Act, requiring all AI systems affecting more than 1 million people to undergo mandatory safety audits."},
    {"id": "E001", "category": "entertainment", "title": "Oscar-Winning AI-Directed Film Sparks Industry Debate", "abstract": "An AI-directed film won the Academy Award for Best Picture, triggering fierce debate about creative authorship, SAG-AFTRA intellectual property rights, and the future of Hollywood."},
    {"id": "SC001", "category": "science", "title": "Researchers Reverse Aging in Human Cells by 25 Years", "abstract": "Harvard scientists demonstrated reversing epigenetic aging markers in human liver cells by 25 years using Yamanaka factor gene therapy, with clinical trials planned for 2027."},
    {"id": "B001", "category": "business", "title": "Apple Becomes First $10 Trillion Market Cap Company", "abstract": "Apple Inc. crossed the $10 trillion market capitalization milestone, driven by record iPhone 17 sales, services growth, and investor optimism about its generative AI integration."},
    {"id": "L001", "category": "lifestyle", "title": "Mindfulness Apps Report 500% Surge in Teen Usage", "abstract": "Mental wellness applications like Calm and Headspace reported a 500% surge in teenage users following school district mandates for daily five-minute mindfulness sessions."},
    {"id": "H001", "category": "health", "title": "GLP-1 Drug Revolution: Obesity Rate Falls Below 20% in US", "abstract": "The widespread adoption of GLP-1 receptor agonists like semaglutide caused the US obesity rate to fall below 20% for the first time in 40 years, with profound implications for healthcare costs."},
]


def build_synthetic_dataset() -> pd.DataFrame:
    df = pd.DataFrame(ARTICLES).rename(columns={"id": "news_id"})
    df["subcategory"] = df["category"].astype(str) + "_feature"
    df["url"] = ""
    df["title_entities"] = [[] for _ in range(len(df))]
    df["abstract_entities"] = [[] for _ in range(len(df))]
    df["entities"] = [[] for _ in range(len(df))]
    df["entity_ids"] = [[] for _ in range(len(df))]
    df["entity_labels"] = [[] for _ in range(len(df))]
    df["source"] = "synthetic"
    df["text"] = df["title"] + ". " + df["abstract"]
    df["impressions_count"] = 0
    df["click_count"] = 0
    df["ctr"] = 0.0
    df["popularity"] = 0.0
    return df


def resolve_input_paths(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.news_path:
        return list(args.news_path), list(args.behaviors_path or [])
    return resolve_default_mind_paths(_BASE)


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    news_paths, behavior_paths = resolve_input_paths(args)
    if news_paths:
        print("📰 Loading MIND news data...")
        for path in news_paths:
            print(f"   news: {path}")
        for path in behavior_paths:
            print(f"   behaviors: {path}")
        df = load_mind_news(news_paths, behavior_paths=behavior_paths)
        if args.max_articles and len(df) > args.max_articles:
            if "popularity" in df.columns:
                df = df.sort_values("popularity", ascending=False).head(args.max_articles).reset_index(drop=True)
            else:
                df = df.head(args.max_articles).reset_index(drop=True)
            print(f"   Trimmed dataset to top {len(df):,} articles for local development.")
        print(f"   Loaded {len(df):,} articles with entity metadata.")
        return df

    print("📰 No raw MIND files found, falling back to synthetic demo data.")
    df = build_synthetic_dataset()
    if args.max_articles and len(df) > args.max_articles:
        df = df.head(args.max_articles).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build HyperNews retrieval assets.")
    parser.add_argument("--news-path", action="append", help="Path to a MIND news.tsv file. Can be provided multiple times.")
    parser.add_argument("--behaviors-path", action="append", help="Path to a MIND behaviors.tsv file. Can be provided multiple times.")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    parser.add_argument("--max-articles", type=int, default=0, help="Optional cap for local/dev builds.")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    df = build_dataset(args)

    print("💾 Writing parquet files...")
    df.to_parquet(PARQUET, index=False)
    df.to_parquet(LEGACY_PARQUET, index=False)
    print(f"   Saved {len(df):,} articles -> {PARQUET}")

    print("🔢 Computing embeddings...")
    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    np.save(EMB_FILE, embeddings)
    print(f"   Saved embeddings {embeddings.shape} -> {EMB_FILE}")

    print("🔍 Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    emb_copy = embeddings.copy()
    faiss.normalize_L2(emb_copy)
    index.add(emb_copy)
    faiss.write_index(index, FAISS_FILE)
    faiss.write_index(index, LEGACY_FAISS_FILE)
    print(f"   Saved FAISS index -> {FAISS_FILE}")
    print("✅ Data pipeline complete.")


if __name__ == "__main__":
    main()
