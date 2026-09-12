"""
Retriever — FAISS-based vector store for retrieving similar historical support interactions.
Used to ground reply generation in how Apple has historically handled similar issues.
"""
import json
import sys
import pickle
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import get_openai_client_kwargs, EMBEDDING_MODEL, EMBEDDINGS_DIR, DATA_DIR, RETRIEVER_TOP_K


class SupportRetriever:
    """
    Vector-based retriever for finding similar historical support interactions.
    
    Uses OpenAI embeddings + FAISS for efficient similarity search.
    Each entry in the index is a (customer_message, agent_reply) pair.
    """
    
    def __init__(self):
        self.client = OpenAI(**get_openai_client_kwargs())
        self.index = None
        self.pairs = []
        self.embeddings = None
        self.index_path = EMBEDDINGS_DIR / "faiss_index.pkl"
        self.pairs_path = EMBEDDINGS_DIR / "indexed_pairs.jsonl"
        self.embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    
    def _get_embeddings(self, texts: list[str], batch_size: int = 50) -> np.ndarray:
        """Get embeddings for a list of texts using OpenAI-compatible API."""
        import time
        all_embeddings = []
        embed_dim = None  # Auto-detect from first successful batch
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            
            # Clean empty strings
            batch = [t if t.strip() else "empty" for t in batch]
            
            # Retry with backoff for rate limiting
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch
                    )
                    batch_embeddings = [e.embedding for e in response.data]
                    if embed_dim is None:
                        embed_dim = len(batch_embeddings[0])
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        wait_time = min(60, 2 ** attempt * 10)
                        print(f"⚠️  Rate limited at batch {i}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"⚠️  Embedding error at batch {i}: {e}")
                        # Fill with zeros for failed batches
                        dim = embed_dim or 3072
                        all_embeddings.extend([np.zeros(dim).tolist()] * len(batch))
                        break
            else:
                # All retries exhausted
                print(f"⚠️  All retries failed at batch {i}, filling with zeros")
                dim = embed_dim or 3072
                all_embeddings.extend([np.zeros(dim).tolist()] * len(batch))
            
            # Rate limit: sleep between batches to stay under free-tier quota
            time.sleep(1.5)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def build_index(self, pairs: list[dict], max_pairs: int = 5000):
        """
        Build the FAISS index from customer-agent pairs.
        
        We embed the customer messages and store the full pairs for retrieval.
        """
        import faiss
        
        print(f"🏗️  Building retrieval index from {len(pairs):,} pairs...")
        
        # Limit pairs for cost/speed
        if len(pairs) > max_pairs:
            import random
            random.seed(42)
            pairs = random.sample(pairs, max_pairs)
        
        self.pairs = pairs
        
        # Get embeddings for customer messages
        customer_messages = [p["customer_message"] for p in pairs]
        self.embeddings = self._get_embeddings(customer_messages)
        
        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"✅ Index built with {self.index.ntotal} vectors (dim={dim})")
        
        # Save index
        self._save()
    
    def _save(self):
        """Save the index, pairs, and embeddings to disk."""
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        import faiss
        faiss.write_index(self.index, str(self.index_path.with_suffix('.faiss')))
        
        # Save pairs
        with open(self.pairs_path, "w") as f:
            for pair in self.pairs:
                f.write(json.dumps(pair) + "\n")
        
        # Save embeddings
        np.save(self.embeddings_path, self.embeddings)
        
        print(f"💾 Index saved to {EMBEDDINGS_DIR}")
    
    def load(self) -> bool:
        """Load a previously built index from disk."""
        import faiss
        
        faiss_path = self.index_path.with_suffix('.faiss')
        
        if not faiss_path.exists() or not self.pairs_path.exists():
            print("⚠️  No saved index found. Build one first.")
            return False
        
        self.index = faiss.read_index(str(faiss_path))
        
        self.pairs = []
        with open(self.pairs_path) as f:
            for line in f:
                self.pairs.append(json.loads(line))
        
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
        
        print(f"✅ Loaded index with {self.index.ntotal} vectors, {len(self.pairs)} pairs")
        return True
    
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve the top-k most similar historical interactions for a query.
        
        Args:
            query: The customer message to find similar examples for
            top_k: Number of results to return (default: RETRIEVER_TOP_K)
        
        Returns:
            List of dicts with keys: customer_message, agent_reply, similarity_score
        """
        import faiss
        
        if self.index is None:
            if not self.load():
                return []
        
        if top_k is None:
            top_k = RETRIEVER_TOP_K
        
        # Get query embedding
        query_embedding = self._get_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.pairs):
                pair = self.pairs[idx]
                results.append({
                    "customer_message": pair["customer_message"],
                    "agent_reply": pair["agent_reply"],
                    "similarity_score": round(float(score), 4),
                    "pair_id": pair.get("pair_id", "")
                })
        
        return results
    
    def retrieve_batch(self, queries: list[str], top_k: int = None) -> list[list[dict]]:
        """Retrieve similar examples for a batch of queries."""
        return [self.retrieve(q, top_k) for q in queries]


if __name__ == "__main__":
    # Quick test
    retriever = SupportRetriever()
    
    # Try loading existing index
    if retriever.load():
        results = retriever.retrieve("My iPhone battery drains really fast")
        print("\n🔍 Query: 'My iPhone battery drains really fast'")
        print(f"   Found {len(results)} similar interactions:")
        for r in results:
            print(f"   [{r['similarity_score']:.3f}] Customer: {r['customer_message'][:80]}...")
            print(f"            Agent: {r['agent_reply'][:80]}...")
            print()
    else:
        print("No index found. Run the pipeline to build one.")
