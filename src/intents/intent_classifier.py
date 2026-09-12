"""
Intent Classifier — Three approaches for classifying customer message intents.

Baseline 1 (Trivial): Keyword/regex matching
Baseline 2 (Simple):  TF-IDF + Logistic Regression
Baseline 3 (Full):    LLM-based classification with structured output
"""
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    get_openai_client_kwargs, CLASSIFIER_MODEL, INTENT_TAXONOMY, INTENT_LIST,
    INTENT_CLASSIFICATION_PROMPT, get_intent_descriptions_for_prompt
)


class KeywordClassifier:
    """
    Trivial Baseline: Classify intents using keyword matching.
    Checks message for keywords defined in the intent taxonomy.
    """
    
    def __init__(self):
        self.name = "keyword_baseline"
        self.taxonomy = INTENT_TAXONOMY
    
    def classify(self, message: str) -> dict:
        """Classify a single message using keyword matching."""
        message_lower = message.lower()
        
        scores = {}
        for intent, info in self.taxonomy.items():
            if intent == "other":
                continue
            
            # Count keyword matches
            count = 0
            matched_keywords = []
            for keyword in info["keywords"]:
                if keyword.lower() in message_lower:
                    count += 1
                    matched_keywords.append(keyword)
            
            if count > 0:
                scores[intent] = {
                    "count": count,
                    "keywords": matched_keywords
                }
        
        if not scores:
            return {
                "intent": "other",
                "confidence": 0.3,
                "reasoning": "No keyword matches found",
                "method": self.name
            }
        
        # Pick the intent with the most keyword matches
        best_intent = max(scores, key=lambda k: scores[k]["count"])
        max_count = scores[best_intent]["count"]
        total_keywords = len(self.taxonomy[best_intent]["keywords"])
        confidence = min(0.9, max_count / max(total_keywords, 1) + 0.1)
        
        return {
            "intent": best_intent,
            "confidence": round(confidence, 2),
            "reasoning": f"Matched keywords: {scores[best_intent]['keywords']}",
            "method": self.name
        }
    
    def classify_batch(self, messages: list[str]) -> list[dict]:
        """Classify a batch of messages."""
        return [self.classify(msg) for msg in messages]


class TFIDFClassifier:
    """
    Simple Baseline: TF-IDF + Logistic Regression classifier.
    Trained on labelled examples from the golden set or pseudo-labels.
    """
    
    def __init__(self):
        self.name = "tfidf_logreg_baseline"
        self.vectorizer = None
        self.model = None
        self.is_trained = False
    
    def train(self, messages: list[str], labels: list[str]):
        """Train the TF-IDF + LogReg model."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        
        print(f"🏋️ Training TF-IDF + LogReg on {len(messages)} examples...")
        
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(messages)
        
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        )
        self.model.fit(X, encoded_labels)
        self.is_trained = True
        
        print(f"✅ Model trained with {len(self.label_encoder.classes_)} classes")
    
    def classify(self, message: str) -> dict:
        """Classify a single message."""
        if not self.is_trained:
            return {
                "intent": "other",
                "confidence": 0.0,
                "reasoning": "Model not trained",
                "method": self.name
            }
        
        X = self.vectorizer.transform([message])
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        intent = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            "intent": intent,
            "confidence": round(float(confidence), 2),
            "reasoning": f"TF-IDF + LogReg prediction (confidence: {confidence:.2f})",
            "method": self.name
        }
    
    def classify_batch(self, messages: list[str]) -> list[dict]:
        """Classify a batch of messages."""
        if not self.is_trained:
            return [self.classify(msg) for msg in messages]
        
        X = self.vectorizer.transform(messages)
        probs = self.model.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        intents = self.label_encoder.inverse_transform(pred_indices)
        confidences = probs[np.arange(len(probs)), pred_indices]
        
        return [
            {
                "intent": intent,
                "confidence": round(float(conf), 2),
                "reasoning": f"TF-IDF + LogReg prediction",
                "method": self.name
            }
            for intent, conf in zip(intents, confidences)
        ]


class LLMClassifier:
    """
    Full Agent: LLM-based intent classification using structured output.
    Uses GPT-4o-mini for cost efficiency with the full intent taxonomy.
    """
    
    def __init__(self, model: str = None):
        self.name = "llm_classifier"
        self.model = model or CLASSIFIER_MODEL
        self.client = OpenAI(**get_openai_client_kwargs())
        self.intent_descriptions = get_intent_descriptions_for_prompt()
    
    def classify(self, message: str) -> dict:
        """Classify a single message using the LLM."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            intent_descriptions=self.intent_descriptions,
            message=message
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate intent is in our taxonomy
            intent = result.get("intent", "other")
            if intent not in INTENT_LIST:
                intent = "other"
            
            return {
                "intent": intent,
                "confidence": round(float(result.get("confidence", 0.5)), 2),
                "reasoning": result.get("reasoning", ""),
                "method": self.name
            }
        
        except Exception as e:
            print(f"⚠️  LLM classification error: {e}")
            return {
                "intent": "other",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "method": self.name
            }
    
    def classify_batch(self, messages: list[str], show_progress: bool = True) -> list[dict]:
        """Classify a batch of messages."""
        import time
        results = []
        iterator = messages
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(messages, desc="LLM classifying")
        
        for msg in iterator:
            results.append(self.classify(msg))
            time.sleep(4.1) # 15 RPM limit
        
        return results


def get_classifier(method: str = "llm") -> KeywordClassifier | TFIDFClassifier | LLMClassifier:
    """
    Factory function to get a classifier by method name.
    
    Args:
        method: "keyword", "tfidf", or "llm"
    """
    classifiers = {
        "keyword": KeywordClassifier,
        "tfidf": TFIDFClassifier,
        "llm": LLMClassifier
    }
    
    if method not in classifiers:
        raise ValueError(f"Unknown classifier method: {method}. Choose from {list(classifiers.keys())}")
    
    return classifiers[method]()


def generate_pseudo_labels(pairs: list[dict], sample_size: int = 1000) -> tuple[list[str], list[str]]:
    """
    Generate pseudo-labels using the LLM classifier for training the TF-IDF model.
    This is used when we don't have enough golden-set labels for supervised training.
    """
    import random
    
    print(f"🏷️ Generating pseudo-labels for {sample_size} messages...")
    
    messages = [p["customer_message"] for p in pairs]
    if len(messages) > sample_size:
        messages = random.sample(messages, sample_size)
    
    llm = LLMClassifier()
    results = llm.classify_batch(messages)
    
    # Filter to high-confidence predictions
    labeled_messages = []
    labeled_intents = []
    
    for msg, result in zip(messages, results):
        if result["confidence"] >= 0.6:  # Only use confident predictions
            labeled_messages.append(msg)
            labeled_intents.append(result["intent"])
    
    print(f"✅ Generated {len(labeled_messages)} high-confidence pseudo-labels")
    
    # Distribution
    from collections import Counter
    dist = Counter(labeled_intents)
    print("📊 Label distribution:")
    for intent, count in dist.most_common():
        print(f"   {intent}: {count} ({count/len(labeled_intents)*100:.1f}%)")
    
    return labeled_messages, labeled_intents


if __name__ == "__main__":
    # Quick test
    test_messages = [
        "My iPhone keeps restarting randomly",
        "I was charged twice for an app",
        "How do I use Screen Time?",
        "I can't sign into my Apple ID",
        "This is terrible service, I'm done with Apple",
        "WiFi keeps dropping on my MacBook",
        "My battery drains in 2 hours since the update"
    ]
    
    print("=" * 60)
    print("Testing Keyword Classifier")
    print("=" * 60)
    kw = KeywordClassifier()
    for msg in test_messages:
        result = kw.classify(msg)
        print(f"  [{result['intent']:25s}] ({result['confidence']:.2f}) {msg}")
    
    print("\n" + "=" * 60)
    print("Testing LLM Classifier")
    print("=" * 60)
    llm = LLMClassifier()
    for msg in test_messages:
        result = llm.classify(msg)
        print(f"  [{result['intent']:25s}] ({result['confidence']:.2f}) {msg}")
