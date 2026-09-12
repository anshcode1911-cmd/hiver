"""
Smoke tests for the pipeline.
Tests basic functionality without requiring API keys or full dataset.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loads():
    """Test that config module loads without errors."""
    from src.config import INTENT_TAXONOMY, INTENT_LIST, get_intent_descriptions_for_prompt
    
    assert len(INTENT_TAXONOMY) == 12, f"Expected 12 intents, got {len(INTENT_TAXONOMY)}"
    assert len(INTENT_LIST) == 12
    assert "other" in INTENT_LIST
    assert "device_troubleshooting" in INTENT_LIST
    
    desc = get_intent_descriptions_for_prompt()
    assert len(desc) > 100, "Intent descriptions should be substantial"
    print("✅ Config loads correctly")


def test_keyword_classifier():
    """Test keyword classifier on known examples."""
    from src.intents.intent_classifier import KeywordClassifier
    
    kw = KeywordClassifier()
    
    # Test clear cases
    result = kw.classify("My iPhone keeps restarting")
    assert result["intent"] == "device_troubleshooting", f"Expected device_troubleshooting, got {result['intent']}"
    
    result = kw.classify("How do I use Screen Time?")
    assert result["intent"] == "feature_inquiry", f"Expected feature_inquiry, got {result['intent']}"
    
    result = kw.classify("I was charged twice")
    assert result["intent"] == "billing_payment", f"Expected billing_payment, got {result['intent']}"
    
    # Test unknown
    result = kw.classify("Hello")
    assert result["intent"] == "other"
    
    print("✅ Keyword classifier works correctly")


def test_rule_based_escalation():
    """Test rule-based escalation on known examples."""
    from src.escalation.escalation import RuleBasedEscalation
    
    engine = RuleBasedEscalation()
    
    # Should escalate
    result = engine.decide("MY ACCOUNT HAS BEEN HACKED!!!", "account_access")
    assert result["decision"] == "escalate", f"Expected escalate, got {result['decision']}"
    
    result = engine.decide("I want a refund for this charge", "billing_payment")
    assert result["decision"] == "escalate"
    
    # Should auto-handle
    result = engine.decide("How do I use Screen Time?", "feature_inquiry")
    assert result["decision"] == "auto_handle"
    
    print("✅ Rule-based escalation works correctly")


def test_text_cleaning():
    """Test text cleaning function."""
    from src.data.preprocess import clean_text
    
    assert clean_text("Check this https://t.co/abc123") == "Check this [URL]"
    assert "@user123" not in clean_text("@user123 help me")
    assert "@AppleSupport" in clean_text("@AppleSupport my phone broke")
    assert clean_text("  multiple   spaces  ") == "multiple spaces"
    assert clean_text("") == ""
    
    print("✅ Text cleaning works correctly")


def test_difficulty_classification():
    """Test difficulty heuristics."""
    from src.data.create_golden_set import classify_difficulty
    
    assert classify_difficulty("How do I use Screen Time on my iPhone?") == "easy"
    assert classify_difficulty("HELP!!!") in ["medium", "hard"]
    assert classify_difficulty("THIS IS RIDICULOUS!!! MY PHONE IS BROKEN AND ALSO MY ICLOUD ISN'T WORKING!!!") == "hard"
    
    print("✅ Difficulty classification works correctly")


def test_escalation_determination():
    """Test escalation ground truth determination."""
    from src.data.create_golden_set import determine_escalation
    
    decision, reason = determine_escalation("My account was hacked", "account_access")
    assert decision == "escalate"
    
    decision, reason = determine_escalation("How do I use AirDrop?", "feature_inquiry")
    assert decision == "auto_handle"
    
    decision, reason = determine_escalation("I want my money back, you charged me twice!", "billing_payment")
    assert decision == "escalate"
    
    print("✅ Escalation determination works correctly")


def test_metrics_computation():
    """Test metrics computation with dummy data."""
    from src.evaluation.metrics import compute_intent_metrics, compute_escalation_metrics
    
    # Intent metrics
    y_true = ["a", "b", "c", "a", "b", "c", "a", "b"]
    y_pred = ["a", "b", "c", "a", "c", "c", "b", "b"]
    
    metrics = compute_intent_metrics(y_true, y_pred)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["weighted_f1"] <= 1
    assert metrics["total_samples"] == 8
    
    # Escalation metrics
    y_true_esc = ["auto_handle", "escalate", "escalate", "auto_handle", "escalate"]
    y_pred_esc = ["auto_handle", "escalate", "auto_handle", "auto_handle", "escalate"]
    
    esc_metrics = compute_escalation_metrics(y_true_esc, y_pred_esc)
    assert 0 <= esc_metrics["accuracy"] <= 1
    assert 0 <= esc_metrics["missed_escalation_rate"] <= 1
    
    print("✅ Metrics computation works correctly")


def test_reply_metrics():
    """Test reply quality metrics."""
    from src.evaluation.metrics import compute_reply_metrics
    
    generated = ["Hello, how can I help you today?", "Please try restarting your device."]
    reference = ["Hi! How can we help?", "Try restarting the device and let us know."]
    
    metrics = compute_reply_metrics(generated, reference)
    assert metrics["num_evaluated"] == 2
    assert metrics["bleu4_mean"] >= 0
    assert metrics["rougeL_mean"] >= 0
    assert metrics["avg_generated_length"] > 0
    
    print("✅ Reply metrics work correctly")


if __name__ == "__main__":
    print("🧪 Running smoke tests...\n")
    
    test_config_loads()
    test_keyword_classifier()
    test_rule_based_escalation()
    test_text_cleaning()
    test_difficulty_classification()
    test_escalation_determination()
    test_metrics_computation()
    test_reply_metrics()
    
    print("\n✅ All smoke tests passed!")
