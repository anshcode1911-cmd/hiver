"""
Download the Customer Support on Twitter dataset from Kaggle.
Filters to the target brand (AppleSupport) and saves a subset.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_DIR, RAW_DATA_DIR, TARGET_BRAND, TARGET_BRAND_ALIASES, MAX_THREADS
)


def download_dataset() -> Path:
    """
    Download the Kaggle dataset using kagglehub.
    Returns the path to the downloaded CSV file.
    """
    try:
        import kagglehub
        print("📥 Downloading dataset from Kaggle via kagglehub...")
        path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        dataset_path = Path(path)
        print(f"✅ Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"⚠️  kagglehub download failed: {e}")
        print("Trying kaggle CLI fallback...")
        return _download_via_cli()


def _download_via_cli() -> Path:
    """Fallback: use kaggle CLI to download."""
    output_dir = RAW_DATA_DIR
    os.system(
        f"kaggle datasets download -d thoughtvector/customer-support-on-twitter "
        f"-p {output_dir} --unzip"
    )
    return output_dir


def find_csv_file(dataset_path: Path) -> Path:
    """Find the main CSV file in the downloaded dataset directory."""
    # Look for common names
    candidates = [
        "twcs.csv",
        "sample.csv",
        "customer_support_twitter.csv",
    ]
    
    # Try known filenames first
    for name in candidates:
        for p in dataset_path.rglob(name):
            return p
    
    # Fall back to any CSV
    csvs = list(dataset_path.rglob("*.csv"))
    if csvs:
        # Pick the largest CSV (likely the main dataset)
        return max(csvs, key=lambda p: p.stat().st_size)
    
    raise FileNotFoundError(f"No CSV files found in {dataset_path}")


def filter_brand(df: pd.DataFrame, brand: str, aliases: list) -> pd.DataFrame:
    """
    Filter the dataset to conversations involving the target brand.
    The brand appears as an author_id for outbound (agent) tweets.
    """
    print(f"🔍 Filtering for brand: {brand}")
    
    # Find all author_ids that match the brand
    # In the dataset, brand names appear in the 'author_id' column
    brand_mask = df["author_id"].astype(str).str.lower().isin(
        [a.lower() for a in aliases]
    )
    
    # Get all tweet_ids from the brand
    brand_tweet_ids = set(df.loc[brand_mask, "tweet_id"].astype(str).values)
    
    # Get all tweets that are responses to brand tweets or that the brand responded to
    brand_response_ids = set(
        df.loc[brand_mask, "in_response_to_tweet_id"].dropna().astype(str).values
    )
    
    # Get all tweets that the brand responded to
    responded_to = set(
        df.loc[brand_mask, "response_tweet_id"].dropna().astype(str).values
    )
    
    # Combine: keep brand tweets + tweets they responded to + tweets responding to them
    all_relevant_ids = brand_tweet_ids | brand_response_ids | responded_to
    
    # Also get inbound tweets that the brand replied to
    inbound_tweet_ids = set(
        df.loc[
            df["tweet_id"].astype(str).isin(brand_response_ids) |
            df["response_tweet_id"].astype(str).isin(brand_tweet_ids),
            "tweet_id"
        ].astype(str).values
    )
    
    all_relevant_ids = all_relevant_ids | inbound_tweet_ids
    
    filtered = df[
        df["tweet_id"].astype(str).isin(all_relevant_ids) | brand_mask
    ].copy()
    
    print(f"✅ Found {len(filtered):,} tweets involving {brand}")
    print(f"   - Brand tweets: {brand_mask.sum():,}")
    print(f"   - Customer tweets: {len(filtered) - brand_mask.sum():,}")
    
    return filtered


def subsample_threads(df: pd.DataFrame, max_threads: int) -> pd.DataFrame:
    """
    Subsample to max_threads conversation threads.
    A thread starts with an inbound customer tweet.
    """
    # Identify thread starters: inbound tweets that are not responses to other tweets
    # or inbound tweets whose in_response_to is a brand tweet
    inbound = df[df["inbound"] == True].copy()
    
    if len(inbound) > max_threads:
        # Stratified sampling by time if 'created_at' exists
        if "created_at" in inbound.columns:
            inbound = inbound.sort_values("created_at")
            # Take evenly spaced samples
            step = len(inbound) // max_threads
            inbound = inbound.iloc[::max(step, 1)][:max_threads]
        else:
            inbound = inbound.sample(n=max_threads, random_state=42)
    
    # Get all tweet IDs in these threads
    thread_tweet_ids = set(inbound["tweet_id"].astype(str).values)
    
    # Add responses to these tweets
    response_ids = set(
        inbound["response_tweet_id"].dropna().astype(str).values
    )
    thread_tweet_ids = thread_tweet_ids | response_ids
    
    # Get the full rows for all tweets in these threads
    result = df[df["tweet_id"].astype(str).isin(thread_tweet_ids)].copy()
    
    print(f"📊 Subsampled to {len(result):,} tweets from ~{len(inbound):,} threads")
    return result


def run_download(output_path: Path = None) -> Path:
    """
    Main download pipeline.
    Returns path to the filtered, subsampled CSV.
    """
    if output_path is None:
        output_path = DATA_DIR / "apple_support_raw.csv"
    
    # Check if already downloaded
    if output_path.exists():
        print(f"✅ Data already exists at {output_path}")
        return output_path
    
    # Download
    dataset_path = download_dataset()
    
    # Find the CSV
    csv_path = find_csv_file(dataset_path)
    print(f"📄 Reading CSV from: {csv_path}")
    
    # Load
    df = pd.read_csv(csv_path)
    print(f"📊 Full dataset: {len(df):,} tweets, {df['author_id'].nunique():,} unique authors")
    
    # Show top brands
    outbound = df[df["inbound"] == False]
    top_brands = outbound["author_id"].value_counts().head(20)
    print("\n🏢 Top 20 brands by tweet volume:")
    for brand, count in top_brands.items():
        marker = " ← TARGET" if str(brand).lower() in [a.lower() for a in TARGET_BRAND_ALIASES] else ""
        print(f"   {brand}: {count:,}{marker}")
    
    # Filter to target brand
    filtered = filter_brand(df, TARGET_BRAND, TARGET_BRAND_ALIASES)
    
    # Subsample
    subsampled = subsample_threads(filtered, MAX_THREADS)
    
    # Save
    subsampled.to_csv(output_path, index=False)
    print(f"\n💾 Saved filtered data to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_path


if __name__ == "__main__":
    run_download()
