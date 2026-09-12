"""
Escalation Decision Engine — Determines whether a customer message should be
auto-handled by the AI agent or escalated to a human support agent.

Three approaches:
1. Trivial baseline: Always escalate (safest, but useless)
2. Simple baseline: Rule-based keyword matching
3. Full agent: Hybrid rules + LLM reasoning
"""
import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    get_openai_client_kwargs, ESCALATION_MODEL, ESCALATION_KEYWORDS, ESCALATION_PROMPT
)


class AlwaysEscalateBaseline:
    """
    Trivial Baseline: Always escalate to a human.
    This is the safest possible policy — it never makes a wrong auto-handle decision,
    but it's useless as an automation system.
    """
    
    def __init__(self):
        self.name = "always_escalate_baseline"
    
    def decide(self, message: str, intent: str = None) -> dict:
        return {
            "decision": "escalate",
            "reason": "Policy: all messages are escalated to human agents",
            "confidence": 1.0,
            "criteria_triggered": ["default_policy"],
            "method": self.name
        }


class RuleBasedEscalation:
    """
    Simple Baseline: Rule-based escalation using keyword matching and intent heuristics.
    """
    
    def __init__(self):
        self.name = "rule_based_baseline"
        self.escalation_keywords = ESCALATION_KEYWORDS
        
        # Intents that should always escalate
        self.always_escalate_intents = {
            "billing_payment",  # Money involved
            "repair_warranty",  # Needs human scheduling
        }
        
        # Intents that should usually auto-handle
        self.usually_auto_intents = {
            "feature_inquiry",
            "connectivity",
            "update_installation",
            "battery_performance",
            "other",
        }
    
    def decide(self, message: str, intent: str = None) -> dict:
        """Decide based on keyword matching and intent heuristics."""
        message_lower = message.lower()
        
        # Check for escalation keywords
        triggered = []
        for keyword in self.escalation_keywords:
            if keyword.lower() in message_lower:
                triggered.append(keyword)
        
        # Check intent-based rules
        if intent in self.always_escalate_intents:
            triggered.append(f"intent:{intent}")
        
        # High emotion detection (simple heuristics)
        exclamation_count = message.count("!")
        caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)
        
        if exclamation_count >= 3:
            triggered.append("high_emotion:exclamations")
        if caps_ratio > 0.5 and len(message) > 10:
            triggered.append("high_emotion:caps")
        
        # Decision
        if triggered:
            return {
                "decision": "escalate",
                "reason": f"Triggered: {', '.join(triggered[:3])}",
                "confidence": min(0.9, 0.5 + 0.1 * len(triggered)),
                "criteria_triggered": triggered,
                "method": self.name
            }
        else:
            return {
                "decision": "auto_handle",
                "reason": "No escalation triggers detected",
                "confidence": 0.7,
                "criteria_triggered": [],
                "method": self.name
            }


class HybridEscalation:
    """
    Full Agent: Hybrid escalation combining rule-based pre-filtering with LLM reasoning.
    
    1. First applies fast rule-based checks for obvious escalation triggers
    2. Then uses LLM for nuanced cases that rules can't handle
    """
    
    def __init__(self, model: str = None):
        self.name = "hybrid_escalation"
        self.model = model or ESCALATION_MODEL
        self.client = OpenAI(**get_openai_client_kwargs())
        self.rule_based = RuleBasedEscalation()
    
    def decide(self, message: str, intent: str = None) -> dict:
        """
        Two-stage escalation decision:
        1. Rule-based pre-filter (fast, catches obvious cases)
        2. LLM reasoning (for nuanced cases)
        """
        # Stage 1: Rule-based pre-filter
        rule_result = self.rule_based.decide(message, intent)
        
        # If rules are very confident about escalation, skip LLM
        if rule_result["decision"] == "escalate" and rule_result["confidence"] >= 0.8:
            rule_result["method"] = f"{self.name}:rules_only"
            return rule_result
        
        # Stage 2: LLM reasoning for nuanced cases
        return self._llm_decide(message, intent)
    
    def _llm_decide(self, message: str, intent: str) -> dict:
        """Use LLM for nuanced escalation decisions."""
        prompt = ESCALATION_PROMPT.format(
            message=message,
            intent=intent or "unknown"
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
            
            # Validate decision
            decision = result.get("decision", "escalate")
            if decision not in ("auto_handle", "escalate"):
                decision = "escalate"  # Safe default
            
            return {
                "decision": decision,
                "reason": result.get("reason", "LLM decision"),
                "confidence": round(float(result.get("confidence", 0.5)), 2),
                "criteria_triggered": result.get("criteria_triggered", []),
                "method": f"{self.name}:llm"
            }
        
        except Exception as e:
            print(f"⚠️  Escalation LLM error: {e}")
            # Fail safe: escalate on error
            return {
                "decision": "escalate",
                "reason": f"LLM error, defaulting to escalation: {str(e)}",
                "confidence": 0.5,
                "criteria_triggered": ["error_fallback"],
                "method": f"{self.name}:error_fallback"
            }
    
    def decide_batch(self, messages: list[str], intents: list[str] = None,
                     show_progress: bool = True) -> list[dict]:
        """Decide escalation for a batch of messages."""
        if intents is None:
            intents = [None] * len(messages)
        
        results = []
        iterator = zip(messages, intents)
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Escalation decisions")
        
        for msg, intent in iterator:
            results.append(self.decide(msg, intent))
            import time; time.sleep(4.1)
        
        return results


def get_escalation_engine(method: str = "hybrid"):
    """
    Factory function to get an escalation engine by method name.
    
    Args:
        method: "always_escalate", "rule_based", or "hybrid"
    """
    engines = {
        "always_escalate": AlwaysEscalateBaseline,
        "rule_based": RuleBasedEscalation,
        "hybrid": HybridEscalation
    }
    
    if method not in engines:
        raise ValueError(f"Unknown method: {method}. Choose from {list(engines.keys())}")
    
    return engines[method]()


if __name__ == "__main__":
    test_messages = [
        ("My iPhone keeps restarting randomly", "device_troubleshooting"),
        ("I was charged $99 and I didn't authorize it!!!", "billing_payment"),
        ("How do I use Screen Time?", "feature_inquiry"),
        ("MY ACCOUNT HAS BEEN HACKED HELP!!!", "account_access"),
        ("Apple is a SCAM, I'm contacting my lawyer", "feedback_complaint"),
        ("WiFi keeps dropping on my MacBook", "connectivity"),
        ("My phone caught FIRE!!!", "device_troubleshooting"),
    ]
    
    print("=" * 70)
    print("Testing Escalation Engines")
    print("=" * 70)
    
    for method_name in ["always_escalate", "rule_based", "hybrid"]:
        engine = get_escalation_engine(method_name)
        print(f"\n--- {method_name} ---")
        for msg, intent in test_messages:
            result = engine.decide(msg, intent)
            emoji = "🚨" if result["decision"] == "escalate" else "🤖"
            print(f"  {emoji} [{result['decision']:12s}] ({result['confidence']:.2f}) {msg[:60]}")
