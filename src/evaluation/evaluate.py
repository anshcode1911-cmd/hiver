"""
Evaluate — Main evaluation orchestrator.

Runs all three pipeline components (intent, reply, escalation) on the golden set
using all three baseline methods, computes metrics, and generates reports.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR, RESULTS_DIR
from src.intents.intent_classifier import (
    KeywordClassifier, TFIDFClassifier, LLMClassifier, generate_pseudo_labels
)
from src.reply.reply_generator import (
    TemplateReplyGenerator, TFIDFReplyGenerator, RAGReplyGenerator
)
from src.reply.retriever import SupportRetriever
from src.escalation.escalation import (
    AlwaysEscalateBaseline, RuleBasedEscalation, HybridEscalation
)
from src.evaluation.metrics import (
    compute_intent_metrics, compute_escalation_metrics, compute_reply_metrics,
    plot_confusion_matrix, save_metrics_report
)
from src.evaluation.llm_judge import LLMJudge


def load_golden_set(path: Path = None) -> list[dict]:
    """Load the golden evaluation set."""
    if path is None:
        path = DATA_DIR / "golden_set.jsonl"
    
    if not path.exists():
        raise FileNotFoundError(
            f"Golden set not found at {path}. "
            "Run the golden set creation first."
        )
    
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"📄 Loaded {len(examples)} golden set examples")
    return examples


def load_pairs(path: Path = None) -> list[dict]:
    """Load the preprocessed pairs."""
    if path is None:
        path = DATA_DIR / "apple_pairs.jsonl"
    
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    print(f"📄 Loaded {len(pairs)} pairs")
    return pairs


def evaluate_intent_classification(golden_set: list[dict], pairs: list[dict]) -> dict:
    """
    Evaluate all three intent classifiers on the golden set.
    """
    print("\n" + "=" * 60)
    print("📊 INTENT CLASSIFICATION EVALUATION")
    print("=" * 60)
    
    messages = [ex["customer_message"] for ex in golden_set]
    true_intents = [ex["ground_truth_intent"] for ex in golden_set]
    
    results = {}
    
    # 1. Keyword Baseline
    print("\n--- Keyword Baseline ---")
    kw = KeywordClassifier()
    kw_preds = kw.classify_batch(messages)
    kw_intents = [p["intent"] for p in kw_preds]
    kw_metrics = compute_intent_metrics(true_intents, kw_intents)
    results["keyword"] = kw_metrics
    print(f"   Accuracy: {kw_metrics['accuracy']:.4f}")
    print(f"   Weighted F1: {kw_metrics['weighted_f1']:.4f}")
    print(f"   Macro F1: {kw_metrics['macro_f1']:.4f}")
    
    # 2. TF-IDF + LogReg Baseline
    print("\n--- TF-IDF + LogReg Baseline ---")
    tfidf = TFIDFClassifier()
    
    # Generate pseudo-labels for training (using LLM on a separate set)
    pseudo_msgs, pseudo_labels = generate_pseudo_labels(pairs, sample_size=20)
    tfidf.train(pseudo_msgs, pseudo_labels)
    
    tfidf_preds = tfidf.classify_batch(messages)
    tfidf_intents = [p["intent"] for p in tfidf_preds]
    tfidf_metrics = compute_intent_metrics(true_intents, tfidf_intents)
    results["tfidf"] = tfidf_metrics
    print(f"   Accuracy: {tfidf_metrics['accuracy']:.4f}")
    print(f"   Weighted F1: {tfidf_metrics['weighted_f1']:.4f}")
    print(f"   Macro F1: {tfidf_metrics['macro_f1']:.4f}")
    
    # 3. LLM Classifier
    print("\n--- LLM Classifier ---")
    llm = LLMClassifier()
    llm_preds = llm.classify_batch(messages)
    llm_intents = [p["intent"] for p in llm_preds]
    llm_metrics = compute_intent_metrics(true_intents, llm_intents)
    results["llm"] = llm_metrics
    print(f"   Accuracy: {llm_metrics['accuracy']:.4f}")
    print(f"   Weighted F1: {llm_metrics['weighted_f1']:.4f}")
    print(f"   Macro F1: {llm_metrics['macro_f1']:.4f}")
    
    # Save confusion matrices
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for method, metrics in results.items():
        import numpy as np
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm, metrics["confusion_labels"],
            f"Intent Classification — {method.upper()}",
            RESULTS_DIR / f"confusion_matrix_{method}.png"
        )
    
    # Store raw predictions for analysis
    results["predictions"] = {
        "keyword": kw_preds,
        "tfidf": tfidf_preds,
        "llm": llm_preds
    }
    
    return results


def evaluate_escalation(golden_set: list[dict], intent_preds: dict) -> dict:
    """
    Evaluate all three escalation engines on the golden set.
    """
    print("\n" + "=" * 60)
    print("📊 ESCALATION EVALUATION")
    print("=" * 60)
    
    messages = [ex["customer_message"] for ex in golden_set]
    true_decisions = [ex["ground_truth_escalation"] for ex in golden_set]
    
    # Use LLM-predicted intents as input to escalation
    llm_intents = [p["intent"] for p in intent_preds.get("llm", [{"intent": "other"}] * len(messages))]
    
    results = {}
    
    # 1. Always Escalate
    print("\n--- Always Escalate Baseline ---")
    always = AlwaysEscalateBaseline()
    always_preds = [always.decide(msg)["decision"] for msg in messages]
    always_metrics = compute_escalation_metrics(true_decisions, always_preds)
    results["always_escalate"] = always_metrics
    print(f"   Accuracy: {always_metrics['accuracy']:.4f}")
    print(f"   Missed escalation rate: {always_metrics['missed_escalation_rate']:.4f}")
    print(f"   False escalation rate: {always_metrics['false_escalation_rate']:.4f}")
    
    # 2. Rule-Based
    print("\n--- Rule-Based Baseline ---")
    rules = RuleBasedEscalation()
    rules_preds = [rules.decide(msg, intent)["decision"] for msg, intent in zip(messages, llm_intents)]
    rules_metrics = compute_escalation_metrics(true_decisions, rules_preds)
    results["rule_based"] = rules_metrics
    print(f"   Accuracy: {rules_metrics['accuracy']:.4f}")
    print(f"   Missed escalation rate: {rules_metrics['missed_escalation_rate']:.4f}")
    print(f"   False escalation rate: {rules_metrics['false_escalation_rate']:.4f}")
    
    # 3. Hybrid (Rules + LLM)
    print("\n--- Hybrid Escalation ---")
    hybrid = HybridEscalation()
    hybrid_preds_raw = hybrid.decide_batch(messages, llm_intents)
    hybrid_preds = [p["decision"] for p in hybrid_preds_raw]
    hybrid_metrics = compute_escalation_metrics(true_decisions, hybrid_preds)
    results["hybrid"] = hybrid_metrics
    print(f"   Accuracy: {hybrid_metrics['accuracy']:.4f}")
    print(f"   Missed escalation rate: {hybrid_metrics['missed_escalation_rate']:.4f}")
    print(f"   False escalation rate: {hybrid_metrics['false_escalation_rate']:.4f}")
    
    # Save confusion matrices
    for method, metrics in results.items():
        if "confusion_matrix" in metrics:
            import numpy as np
            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(
                cm, metrics["confusion_labels"],
                f"Escalation — {method.upper()}",
                RESULTS_DIR / f"escalation_cm_{method}.png"
            )
    
    return results


def evaluate_reply_quality(golden_set: list[dict], pairs: list[dict]) -> dict:
    """
    Evaluate all three reply generators on the golden set.
    """
    print("\n" + "=" * 60)
    print("📊 REPLY QUALITY EVALUATION")
    print("=" * 60)
    
    messages = [ex["customer_message"] for ex in golden_set]
    reference_replies = [ex.get("reference_reply", "") for ex in golden_set]
    intents = [ex["ground_truth_intent"] for ex in golden_set]
    
    results = {}
    
    # 1. Template Baseline
    print("\n--- Template Baseline ---")
    template_gen = TemplateReplyGenerator()
    
    # Build templates using LLM-classified pairs
    llm_classifier = LLMClassifier()
    # Use a small sample to build templates
    import random
    sample_pairs = random.sample(pairs, min(20, len(pairs)))
    sample_labels = llm_classifier.classify_batch(
        [p["customer_message"] for p in sample_pairs], 
        show_progress=True
    )
    template_gen.build_templates(sample_pairs, sample_labels)
    
    template_replies = [template_gen.generate(msg, intent)["reply"] for msg, intent in zip(messages, intents)]
    template_metrics = compute_reply_metrics(template_replies, reference_replies)
    results["template"] = template_metrics
    print(f"   BLEU-4: {template_metrics['bleu4_mean']:.4f}")
    print(f"   ROUGE-L: {template_metrics['rougeL_mean']:.4f}")
    
    # 2. TF-IDF Retrieval Baseline
    print("\n--- TF-IDF Retrieval Baseline ---")
    tfidf_gen = TFIDFReplyGenerator()
    tfidf_gen.build_index(pairs)
    
    tfidf_replies = [tfidf_gen.generate(msg)["reply"] for msg in messages]
    tfidf_metrics = compute_reply_metrics(tfidf_replies, reference_replies)
    results["tfidf"] = tfidf_metrics
    print(f"   BLEU-4: {tfidf_metrics['bleu4_mean']:.4f}")
    print(f"   ROUGE-L: {tfidf_metrics['rougeL_mean']:.4f}")
    
    # 3. RAG + LLM Generator
    print("\n--- RAG + LLM Generator ---")
    retriever = SupportRetriever()
    
    # Build or load FAISS index
    if not retriever.load():
        retriever.build_index(pairs, max_pairs=3000)
    
    rag_gen = RAGReplyGenerator(retriever=retriever)
    rag_replies_raw = rag_gen.generate_batch(messages, intents)
    rag_replies = [r["reply"] for r in rag_replies_raw]
    rag_metrics = compute_reply_metrics(rag_replies, reference_replies)
    results["rag"] = rag_metrics
    print(f"   BLEU-4: {rag_metrics['bleu4_mean']:.4f}")
    print(f"   ROUGE-L: {rag_metrics['rougeL_mean']:.4f}")
    
    # Store generated replies for LLM-judge evaluation
    results["generated_replies"] = {
        "template": template_replies,
        "tfidf": tfidf_replies,
        "rag": rag_replies
    }
    
    return results


def evaluate_with_llm_judge(golden_set: list[dict], 
                            generated_replies: list[str],
                            method_name: str = "rag") -> dict:
    """
    Evaluate generated replies using the LLM-as-judge.
    Only run on the RAG replies (the full agent) to save API costs.
    """
    print("\n" + "=" * 60)
    print(f"📊 LLM-AS-JUDGE EVALUATION ({method_name})")
    print("=" * 60)
    
    judge = LLMJudge()
    
    messages = [ex["customer_message"] for ex in golden_set]
    references = [ex.get("reference_reply", "") for ex in golden_set]
    
    evaluations = judge.evaluate_batch(messages, generated_replies, references)
    aggregate = judge.compute_aggregate_scores(evaluations)
    
    print(f"\n   Average Overall Score: {aggregate['avg_overall_score']:.2f}/5.00")
    print(f"   Score Distribution:")
    for bucket, count in aggregate["score_distribution"].items():
        print(f"      {bucket}: {count}")
    
    print(f"\n   Per-Dimension Means:")
    for dim, stats in aggregate["dimension_stats"].items():
        print(f"      {dim:15s}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    return {
        "evaluations": evaluations,
        "aggregate": aggregate
    }


def run_full_evaluation():
    """
    Run the complete evaluation pipeline:
    1. Load golden set and pairs
    2. Evaluate intent classification (3 methods)
    3. Evaluate escalation (3 methods)
    4. Evaluate reply quality (3 methods + LLM judge)
    5. Save all results
    """
    start_time = time.time()
    
    print("🚀 Starting Full Evaluation Pipeline")
    print("=" * 60)
    
    # Load data
    golden_set = load_golden_set()
    pairs = load_pairs()
    
    # 1. Intent Classification
    intent_results = evaluate_intent_classification(golden_set, pairs)
    
    # 2. Escalation
    escalation_results = evaluate_escalation(golden_set, intent_results.get("predictions", {}))
    
    # 3. Reply Quality
    reply_results = evaluate_reply_quality(golden_set, pairs)
    
    # 4. LLM Judge (on RAG replies only to save cost)
    rag_replies = reply_results.get("generated_replies", {}).get("rag", [])
    judge_results = None
    if rag_replies:
        judge_results = evaluate_with_llm_judge(golden_set, rag_replies, "rag")
    
    # 5. Save comprehensive report
    report = save_metrics_report(
        intent_metrics=intent_results.get("llm", {}),
        escalation_metrics=escalation_results.get("hybrid", {}),
        reply_metrics=reply_results.get("rag", {}),
        judge_metrics=judge_results.get("aggregate", {}) if judge_results else None
    )
    
    # Save all detailed results
    all_results = {
        "intent": {k: v for k, v in intent_results.items() if k != "predictions"},
        "escalation": escalation_results,
        "reply": {k: v for k, v in reply_results.items() if k != "generated_replies"},
        "judge": judge_results.get("aggregate", {}) if judge_results else {}
    }
    
    with open(RESULTS_DIR / "full_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save individual predictions for failure analysis
    predictions_path = RESULTS_DIR / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for i, ex in enumerate(golden_set):
            pred = {
                "id": ex.get("id", f"gs_{i:03d}"),
                "customer_message": ex["customer_message"],
                "true_intent": ex["ground_truth_intent"],
                "true_escalation": ex["ground_truth_escalation"],
                "reference_reply": ex.get("reference_reply", ""),
            }
            
            # Add predictions from each method
            for method in ["keyword", "tfidf", "llm"]:
                preds = intent_results.get("predictions", {}).get(method, [])
                if i < len(preds):
                    pred[f"pred_intent_{method}"] = preds[i]["intent"]
            
            for method in ["template", "tfidf", "rag"]:
                replies = reply_results.get("generated_replies", {}).get(method, [])
                if i < len(replies):
                    pred[f"pred_reply_{method}"] = replies[i]
            
            if judge_results and i < len(judge_results.get("evaluations", [])):
                pred["judge_scores"] = judge_results["evaluations"][i].get("scores", {})
                pred["judge_overall"] = judge_results["evaluations"][i].get("overall_score", 0)
            
            f.write(json.dumps(pred) + "\n")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total evaluation time: {elapsed/60:.1f} minutes")
    print(f"💾 All results saved to {RESULTS_DIR}")
    
    return report


if __name__ == "__main__":
    run_full_evaluation()
