import pandas as pd
import numpy as np
import os
import uuid
from sentence_transformers import SentenceTransformer

def generate_synthetic_news(num_articles: int = 150):
    categories = ["Technology", "Politics", "Sports", "Entertainment", "Business", "Science", "World", "Lifestyle"]
    titles = [
        "New AI Model Beats Human Baselines",
        "Elections Approach with High Voter Turnout",
        "Local Team Wins Championship in Overtime",
        "Blockbuster Movie Breaks Box Office Records",
        "Stock Market Reaches All-Time High",
        "Scientists Discover New Exoplanet",
        "Global Summit Addresses Climate Change",
        "Top 10 Healthy Habits for Productivity",
        "Tech Giant Unveils Revolutionary Device",
        "Debate Heats Up Over Proposed Legislation",
        "Star Athlete Signs Record-Breaking Contract",
        "Award Show Celebrates Industry Excellence",
        "Startup Secures $50M in Series B Funding",
        "Breakthrough in Renewable Energy Tech",
        "International Deal Aims to Boost Trade",
        "Minimalist Design Trends for the New Year"
    ]
    abstracts = [
        "A detailed look into the implications of recent developments and what it means for the future.",
        "Experts weigh in on the controversial topic, highlighting both potential benefits and risks.",
        "Fans celebrate an unprecedented victory that has been decades in the making.",
        "Critics are raving about the stunning visuals and compelling narrative of this new release.",
        "Investors are optimistic as economic indicators show strong signs of sustained growth.",
        "A team of international researchers published groundbreaking findings in this month's journal.",
        "Leaders from around the globe gathered to discuss strategies for sustainable development.",
        "Simple lifestyle changes that promise to have a profound impact on daily well-being."
    ]

    data = []
    np.random.seed(42) # for reproducibility
    for _ in range(num_articles):
        cat = np.random.choice(categories)
        title = np.random.choice(titles)
        abs_text = np.random.choice(abstracts)
        
        # Add some variation to make them unique
        uniq = str(uuid.uuid4())[:4]
        
        data.append({
            "news_id": f"N{np.random.randint(10000, 99999)}",
            "category": cat,
            "subcategory": cat + "_sub",
            "title": f"{title} ({uniq})",
            "abstract": abs_text
        })

    df = pd.DataFrame(data)
    df["text"] = df["title"] + " " + df["abstract"]
    
    # Save
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    df.to_parquet(os.path.join(data_dir, "articles.parquet"))
    df.to_parquet(os.path.join(data_dir, "news_processed.parquet"))
    
    # Generate embeddings
    print("Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(os.path.join(data_dir, "article_embeddings.npy"), embeddings)
    
    print(f"Successfully generated {num_articles} synthetic articles and embeddings.")

if __name__ == "__main__":
    generate_synthetic_news()
