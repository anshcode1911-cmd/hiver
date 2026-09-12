"""
Metrics — Automated evaluation metrics for all three pipeline components.

Covers:
1. Intent classification: Accuracy, F1 (weighted/macro), per-class precision/recall
2. Escalation: Binary classification metrics + cost-aware metrics
3. Reply quality: BLEU, ROUGE-L, length analysis
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RESULTS_DIR, INTENT_LIST


def compute_intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Compute intent classification metrics.
    
    Returns:
        Dict with accuracy, weighted_f1, macro_f1, per-class metrics, confusion matrix
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support,
        classification_report, confusion_matrix
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i])
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Classification report (text)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    
    return {
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "classification_report": report,
        "total_samples": len(y_true),
        "label_distribution": dict(Counter(y_true))
    }


def compute_escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Compute escalation decision metrics.
    
    Special focus on:
    - False escalation rate (cost: unnecessary human involvement)
    - Missed escalation rate (risk: AI handles what should be escalated)
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Binary metrics (escalate = positive class)
    labels = ["auto_handle", "escalate"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Extract specific counts
    tn = int(cm[0][0])  # Correct auto_handle
    fp = int(cm[0][1])  # False escalation (was auto, predicted escalate)
    fn = int(cm[1][0])  # Missed escalation (was escalate, predicted auto) — DANGEROUS
    tp = int(cm[1][1])  # Correct escalation
    
    total = tn + fp + fn + tp
    
    false_escalation_rate = fp / max(fp + tn, 1)  # Of auto-handle cases, how many wrongly escalated
    missed_escalation_rate = fn / max(fn + tp, 1)  # Of escalation cases, how many wrongly auto-handled
    
    return {
        "accuracy": round(accuracy, 4),
        "escalation_precision": round(float(precision[1]), 4),
        "escalation_recall": round(float(recall[1]), 4),
        "escalation_f1": round(float(f1[1]), 4),
        "auto_handle_precision": round(float(precision[0]), 4),
        "auto_handle_recall": round(float(recall[0]), 4),
        "auto_handle_f1": round(float(f1[0]), 4),
        "false_escalation_rate": round(false_escalation_rate, 4),
        "missed_escalation_rate": round(missed_escalation_rate, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_samples": total,
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0)
    }


def compute_reply_metrics(generated_replies: list[str], reference_replies: list[str]) -> dict:
    """
    Compute reply quality metrics using automated text comparison.
    
    Metrics:
    - BLEU-4: N-gram overlap (precision-focused)
    - ROUGE-L: Longest common subsequence (recall-focused)
    - Length statistics
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    
    # Initialize scorers
    smoother = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    bleu_scores = []
    rouge_scores = []
    gen_lengths = []
    ref_lengths = []
    
    for gen, ref in zip(generated_replies, reference_replies):
        if not gen or not ref:
            continue
        
        # BLEU
        ref_tokens = ref.lower().split()
        gen_tokens = gen.lower().split()
        
        try:
            bleu = sentence_bleu(
                [ref_tokens], gen_tokens,
                smoothing_function=smoother
            )
        except Exception:
            bleu = 0.0
        bleu_scores.append(bleu)
        
        # ROUGE-L
        rouge_result = scorer.score(ref, gen)
        rouge_scores.append(rouge_result["rougeL"].fmeasure)
        
        # Lengths
        gen_lengths.append(len(gen))
        ref_lengths.append(len(ref))
    
    return {
        "bleu4_mean": round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0,
        "bleu4_median": round(float(np.median(bleu_scores)), 4) if bleu_scores else 0.0,
        "bleu4_std": round(float(np.std(bleu_scores)), 4) if bleu_scores else 0.0,
        "rougeL_mean": round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0,
        "rougeL_median": round(float(np.median(rouge_scores)), 4) if rouge_scores else 0.0,
        "rougeL_std": round(float(np.std(rouge_scores)), 4) if rouge_scores else 0.0,
        "avg_generated_length": round(float(np.mean(gen_lengths)), 1) if gen_lengths else 0.0,
        "avg_reference_length": round(float(np.mean(ref_lengths)), 1) if ref_lengths else 0.0,
        "length_ratio": round(
            float(np.mean(gen_lengths)) / max(float(np.mean(ref_lengths)), 1), 2
        ) if gen_lengths else 0.0,
        "num_evaluated": len(bleu_scores)
    }


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], 
                          title: str, output_path: Path):
    """Save a confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels) * 0.7)))
    
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax
    )
    
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Saved confusion matrix to {output_path}")


def save_metrics_report(intent_metrics: dict, escalation_metrics: dict, 
                        reply_metrics: dict, judge_metrics: dict = None,
                        output_path: Path = None):
    """Save all metrics to a structured JSON report."""
    if output_path is None:
        output_path = RESULTS_DIR / "metrics_report.json"
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report = {
        "intent_classification": intent_metrics,
        "escalation": escalation_metrics,
        "reply_quality_automated": reply_metrics,
    }
    
    if judge_metrics:
        report["reply_quality_llm_judge"] = judge_metrics
    
    # Compute headline numbers
    report["headline"] = {
        "intent_accuracy": intent_metrics.get("accuracy", 0),
        "intent_weighted_f1": intent_metrics.get("weighted_f1", 0),
        "escalation_accuracy": escalation_metrics.get("accuracy", 0),
        "missed_escalation_rate": escalation_metrics.get("missed_escalation_rate", 0),
        "bleu4": reply_metrics.get("bleu4_mean", 0),
        "rougeL": reply_metrics.get("rougeL_mean", 0),
    }
    
    if judge_metrics:
        report["headline"]["llm_judge_avg_score"] = judge_metrics.get("avg_overall_score", 0)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Full metrics report saved to {output_path}")
    
    # Print headline summary
    print("\n" + "=" * 60)
    print("📊 HEADLINE RESULTS")
    print("=" * 60)
    for k, v in report["headline"].items():
        print(f"   {k:30s}: {v}")
    
    return report
