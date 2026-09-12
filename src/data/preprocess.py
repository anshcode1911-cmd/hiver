"""
Preprocess the raw Apple Support Twitter data.
- Reconstruct conversation threads
- Extract (customer_message, agent_reply) pairs
- Clean text
- Output structured JSONL files
"""
import re
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR


def clean_text(text: str) -> str:
    """
    Clean a tweet for NLP processing.
    - Remove URLs
    - Remove @mentions (but keep @AppleSupport for context)
    - Normalize whitespace
    - Handle common encoding issues
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"www\.\S+", "[URL]", text)
    
    # Remove @mentions EXCEPT @AppleSupport
    text = re.sub(r"@(?!AppleSupport)\w+", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove leading/trailing punctuation artifacts
    text = text.strip("- ").strip()
    
    return text


def reconstruct_threads(df: pd.DataFrame) -> list[dict]:
    """
    Reconstruct conversation threads from the flat tweet data.
    
    A thread is a sequence of tweets linked by response_tweet_id and 
    in_response_to_tweet_id. Each thread should have at least one 
    customer message and one agent reply.
    
    Returns a list of thread dicts with structure:
    {
        "thread_id": str,
        "messages": [
            {"role": "customer"|"agent", "text": str, "tweet_id": str, "created_at": str},
            ...
        ]
    }
    """
    print("🔗 Reconstructing conversation threads...")
    
    # Build lookup maps
    tweet_by_id = {}
    responses_to = defaultdict(list)  # tweet_id -> list of response tweet_ids
    
    for _, row in df.iterrows():
        tid = str(row["tweet_id"])
        tweet_by_id[tid] = row
        
        # Map: this tweet's response_tweet_id tells us what tweets respond to it
        if pd.notna(row.get("response_tweet_id")):
            # response_tweet_id is the ID of the tweet that responds to this one
            resp_id = str(row["response_tweet_id"])
            responses_to[tid].append(resp_id)
        
        if pd.notna(row.get("in_response_to_tweet_id")):
            parent_id = str(row["in_response_to_tweet_id"])
            responses_to[parent_id].append(tid)
    
    # Deduplicate response lists
    for k in responses_to:
        responses_to[k] = list(set(responses_to[k]))
    
    # Find thread roots: inbound tweets that are NOT responses to another tweet in the dataset
    roots = []
    for _, row in df.iterrows():
        if row.get("inbound") == True:
            parent_id = row.get("in_response_to_tweet_id")
            if pd.isna(parent_id) or str(parent_id) not in tweet_by_id:
                roots.append(str(row["tweet_id"]))
    
    print(f"   Found {len(roots):,} thread roots")
    
    # BFS to build threads
    threads = []
    visited = set()
    
    for root_id in tqdm(roots, desc="Building threads"):
        if root_id in visited or root_id not in tweet_by_id:
            continue
        
        thread_messages = []
        queue = [root_id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited or current_id not in tweet_by_id:
                continue
            visited.add(current_id)
            
            row = tweet_by_id[current_id]
            role = "customer" if row.get("inbound") == True else "agent"
            text = clean_text(str(row.get("text", "")))
            
            if text:  # Skip empty messages
                thread_messages.append({
                    "role": role,
                    "text": text,
                    "tweet_id": current_id,
                    "created_at": str(row.get("created_at", "")),
                    "author_id": str(row.get("author_id", ""))
                })
            
            # Add responses to queue
            for resp_id in responses_to.get(current_id, []):
                if resp_id not in visited:
                    queue.append(resp_id)
        
        if len(thread_messages) >= 2:  # Need at least customer + agent
            # Sort by created_at if available
            thread_messages.sort(key=lambda m: m.get("created_at", ""))
            
            threads.append({
                "thread_id": root_id,
                "messages": thread_messages,
                "num_messages": len(thread_messages)
            })
    
    print(f"✅ Reconstructed {len(threads):,} threads with ≥2 messages")
    return threads


def extract_pairs(threads: list[dict]) -> list[dict]:
    """
    Extract (customer_message, agent_reply) pairs from threads.
    
    Each pair represents a customer message followed by the agent's response.
    This is the primary unit for training and evaluation.
    """
    print("📝 Extracting customer-agent pairs...")
    
    pairs = []
    pair_id = 0
    
    for thread in threads:
        messages = thread["messages"]
        
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "customer" and messages[i+1]["role"] == "agent":
                # Get conversation context (previous messages in thread)
                context = []
                for j in range(max(0, i-2), i):
                    context.append({
                        "role": messages[j]["role"],
                        "text": messages[j]["text"]
                    })
                
                pair = {
                    "pair_id": f"pair_{pair_id:05d}",
                    "thread_id": thread["thread_id"],
                    "customer_message": messages[i]["text"],
                    "agent_reply": messages[i+1]["text"],
                    "customer_tweet_id": messages[i]["tweet_id"],
                    "agent_tweet_id": messages[i+1]["tweet_id"],
                    "context": context,
                    "customer_author": messages[i].get("author_id", ""),
                    "agent_author": messages[i+1].get("author_id", ""),
                    "created_at": messages[i].get("created_at", "")
                }
                
                # Basic quality filters
                if len(pair["customer_message"]) < 5:
                    continue  # Skip very short messages
                if len(pair["agent_reply"]) < 10:
                    continue  # Skip very short replies
                if pair["customer_message"] == "[URL]":
                    continue  # Skip URL-only messages
                
                pairs.append(pair)
                pair_id += 1
    
    print(f"✅ Extracted {len(pairs):,} customer-agent pairs")
    return pairs


def compute_stats(pairs: list[dict]) -> dict:
    """Compute summary statistics for the extracted pairs."""
    customer_lengths = [len(p["customer_message"]) for p in pairs]
    reply_lengths = [len(p["agent_reply"]) for p in pairs]
    
    stats = {
        "total_pairs": len(pairs),
        "avg_customer_msg_length": sum(customer_lengths) / len(customer_lengths) if customer_lengths else 0,
        "avg_agent_reply_length": sum(reply_lengths) / len(reply_lengths) if reply_lengths else 0,
        "max_customer_msg_length": max(customer_lengths) if customer_lengths else 0,
        "max_agent_reply_length": max(reply_lengths) if reply_lengths else 0,
        "pairs_with_context": sum(1 for p in pairs if len(p["context"]) > 0),
        "unique_threads": len(set(p["thread_id"] for p in pairs))
    }
    
    return stats


def run_preprocess(input_path: Path = None, output_path: Path = None) -> Path:
    """
    Main preprocessing pipeline.
    Returns path to the output JSONL file.
    """
    if input_path is None:
        input_path = DATA_DIR / "apple_support_raw.csv"
    if output_path is None:
        output_path = DATA_DIR / "apple_pairs.jsonl"
    
    # Check if already processed
    if output_path.exists():
        print(f"✅ Preprocessed data already exists at {output_path}")
        return output_path
    
    # Load raw data
    print(f"📄 Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   {len(df):,} tweets loaded")
    
    # Reconstruct threads
    threads = reconstruct_threads(df)
    
    # Save threads
    threads_path = DATA_DIR / "apple_threads.jsonl"
    with open(threads_path, "w") as f:
        for thread in threads:
            f.write(json.dumps(thread) + "\n")
    print(f"💾 Saved {len(threads):,} threads to {threads_path}")
    
    # Extract pairs
    pairs = extract_pairs(threads)
    
    # Compute and display stats
    stats = compute_stats(pairs)
    print("\n📊 Dataset Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.1f}")
        else:
            print(f"   {k}: {v:,}")
    
    # Save stats
    stats_path = DATA_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save pairs
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"\n💾 Saved {len(pairs):,} pairs to {output_path}")
    return output_path


if __name__ == "__main__":
    run_preprocess()
