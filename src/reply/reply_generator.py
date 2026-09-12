"""
Reply Generator — RAG-based reply generation grounded in historical Apple Support responses.

Uses retrieved similar interactions as context to generate replies that match
Apple's actual support style and practices.
"""
import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import get_openai_client_kwargs, REPLY_MODEL, REPLY_GENERATION_PROMPT, MAX_REPLY_LENGTH
from src.reply.retriever import SupportRetriever


class TemplateReplyGenerator:
    """
    Trivial Baseline: Return the most common historical reply for each intent.
    """
    
    def __init__(self):
        self.name = "template_baseline"
        self.templates = {}
    
    def build_templates(self, pairs: list[dict], labels: list[dict]):
        """
        Build templates by finding the most common agent reply per intent.
        """
        from collections import Counter, defaultdict
        
        intent_replies = defaultdict(list)
        for pair, label in zip(pairs, labels):
            intent = label.get("intent", "other")
            intent_replies[intent].append(pair["agent_reply"])
        
        for intent, replies in intent_replies.items():
            # Pick the most "representative" reply (closest to median length)
            if replies:
                median_len = sorted(len(r) for r in replies)[len(replies) // 2]
                self.templates[intent] = min(replies, key=lambda r: abs(len(r) - median_len))
        
        # Default template
        self.templates["other"] = (
            "Hi there! We'd like to help. Could you tell us more about the issue "
            "you're experiencing? We're here for you."
        )
        
        print(f"📋 Built templates for {len(self.templates)} intents")
    
    def generate(self, message: str, intent: str, **kwargs) -> dict:
        """Return the template reply for the given intent."""
        reply = self.templates.get(intent, self.templates.get("other", ""))
        
        return {
            "reply": reply[:MAX_REPLY_LENGTH],
            "method": self.name,
            "grounding": "template_match"
        }


class TFIDFReplyGenerator:
    """
    Simple Baseline: Retrieve the most similar historical reply using TF-IDF similarity.
    No LLM involved — just returns the agent reply from the most similar past interaction.
    """
    
    def __init__(self):
        self.name = "tfidf_retrieval_baseline"
        self.vectorizer = None
        self.reply_matrix = None
        self.replies = []
        self.messages = []
    
    def build_index(self, pairs: list[dict]):
        """Build TF-IDF index over customer messages."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.messages = [p["customer_message"] for p in pairs]
        self.replies = [p["agent_reply"] for p in pairs]
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.reply_matrix = self.vectorizer.fit_transform(self.messages)
        
        print(f"📋 Built TF-IDF index over {len(self.messages)} messages")
    
    def generate(self, message: str, intent: str = None, **kwargs) -> dict:
        """Find the most similar message and return its agent reply."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        if self.vectorizer is None:
            return {
                "reply": "I'd be happy to help! Could you provide more details?",
                "method": self.name,
                "grounding": "default"
            }
        
        query_vec = self.vectorizer.transform([message])
        similarities = cosine_similarity(query_vec, self.reply_matrix)[0]
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        return {
            "reply": self.replies[best_idx][:MAX_REPLY_LENGTH],
            "method": self.name,
            "grounding": f"tfidf_similarity={best_score:.3f}",
            "similar_message": self.messages[best_idx][:100]
        }


class RAGReplyGenerator:
    """
    Full Agent: RAG-based reply generation using FAISS retrieval + LLM generation.
    
    Retrieves similar historical interactions, then generates a new reply
    grounded in Apple's actual support patterns.
    """
    
    def __init__(self, retriever: SupportRetriever = None, model: str = None):
        self.name = "rag_llm_generator"
        self.model = model or REPLY_MODEL
        self.client = OpenAI(**get_openai_client_kwargs())
        self.retriever = retriever or SupportRetriever()
    
    def generate(self, message: str, intent: str, top_k: int = 5) -> dict:
        """
        Generate a reply using RAG:
        1. Retrieve similar past interactions
        2. Use them as grounding context for the LLM
        3. Generate a new reply matching Apple's style
        """
        # Step 1: Retrieve similar examples
        retrieved = self.retriever.retrieve(message, top_k=top_k)
        
        # Format retrieved examples for the prompt
        examples_text = ""
        for i, ex in enumerate(retrieved, 1):
            examples_text += (
                f"\n--- Example {i} (similarity: {ex['similarity_score']:.2f}) ---\n"
                f"Customer: {ex['customer_message']}\n"
                f"Apple Reply: {ex['agent_reply']}\n"
            )
        
        if not examples_text:
            examples_text = "(No similar examples found)"
        
        # Step 2: Generate reply with LLM
        prompt = REPLY_GENERATION_PROMPT.format(
            intent=intent,
            message=message,
            retrieved_examples=examples_text
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Truncate to Twitter limit
            if len(reply) > MAX_REPLY_LENGTH:
                reply = reply[:MAX_REPLY_LENGTH - 3] + "..."
            
            return {
                "reply": reply,
                "method": self.name,
                "grounding": f"rag_top_{top_k}",
                "retrieved_count": len(retrieved),
                "top_similarity": retrieved[0]["similarity_score"] if retrieved else 0.0
            }
        
        except Exception as e:
            print(f"⚠️  Reply generation error: {e}")
            # Fallback to best retrieved reply
            if retrieved:
                return {
                    "reply": retrieved[0]["agent_reply"][:MAX_REPLY_LENGTH],
                    "method": "rag_fallback",
                    "grounding": "retrieval_only",
                    "error": str(e)
                }
            return {
                "reply": "We'd like to help! Please DM us with more details about your issue.",
                "method": "default_fallback",
                "grounding": "none",
                "error": str(e)
            }
    
    def generate_batch(self, messages: list[str], intents: list[str], 
                       show_progress: bool = True) -> list[dict]:
        """Generate replies for a batch of messages."""
        results = []
        iterator = zip(messages, intents)
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Generating replies")
        
        for msg, intent in iterator:
            results.append(self.generate(msg, intent))
        
        return results


def get_reply_generator(method: str = "rag", retriever: SupportRetriever = None):
    """
    Factory function to get a reply generator by method name.
    
    Args:
        method: "template", "tfidf", or "rag"
        retriever: Pre-built SupportRetriever instance (for "rag" method)
    """
    generators = {
        "template": TemplateReplyGenerator,
        "tfidf": TFIDFReplyGenerator,
        "rag": lambda: RAGReplyGenerator(retriever=retriever)
    }
    
    if method not in generators:
        raise ValueError(f"Unknown generator method: {method}. Choose from {list(generators.keys())}")
    
    gen = generators[method]
    return gen() if callable(gen) and not isinstance(gen, type) else gen()
