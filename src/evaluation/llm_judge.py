"""
LLM-as-Judge — Uses GPT-4o to evaluate reply quality on a multi-dimensional rubric.

Includes:
- Structured rubric evaluation (5 dimensions, 1-5 scale)
- Human-judge agreement calibration
- Bias mitigation via chain-of-thought reasoning
"""
import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import get_openai_client_kwargs, JUDGE_MODEL, LLM_JUDGE_PROMPT, RESULTS_DIR


class LLMJudge:
    """
    GPT-4o based judge for evaluating reply quality.
    
    Evaluates on 5 dimensions:
    1. Relevance - Does it address the issue?
    2. Grounding - Is it factually correct?
    3. Tone - Does it match Apple's voice?
    4. Actionability - Clear next steps?
    5. Completeness - Covers all aspects?
    """
    
    DIMENSIONS = ["relevance", "grounding", "tone", "actionability", "completeness"]
    
    def __init__(self, model: str = None):
        self.model = model or JUDGE_MODEL
        self.client = OpenAI(**get_openai_client_kwargs())
    
    def evaluate_single(self, customer_message: str, generated_reply: str,
                        reference_reply: str = "") -> dict:
        """
        Evaluate a single generated reply.
        
        Args:
            customer_message: The customer's original message
            generated_reply: The AI-generated reply to evaluate
            reference_reply: The actual Apple reply (if available)
        
        Returns:
            Dict with scores, reasoning, and overall score
        """
        prompt = LLM_JUDGE_PROMPT.format(
            customer_message=customer_message,
            generated_reply=generated_reply,
            reference_reply=reference_reply or "(No reference available)"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=800
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Extract scores
            scores = result.get("scores", {})
            for dim in self.DIMENSIONS:
                if dim not in scores:
                    scores[dim] = 3  # Default to neutral
                scores[dim] = max(1, min(5, int(scores[dim])))
            
            # Compute overall score (weighted average)
            weights = {
                "relevance": 0.25,
                "grounding": 0.25,
                "tone": 0.15,
                "actionability": 0.20,
                "completeness": 0.15
            }
            overall = sum(scores[d] * weights[d] for d in self.DIMENSIONS)
            
            return {
                "scores": scores,
                "overall_score": round(overall, 2),
                "reasoning": result.get("reasoning", {}),
                "major_issues": result.get("major_issues", []),
                "error": None
            }
        
        except Exception as e:
            print(f"⚠️  Judge error: {e}")
            return {
                "scores": {d: 0 for d in self.DIMENSIONS},
                "overall_score": 0,
                "reasoning": {},
                "major_issues": [f"Evaluation error: {str(e)}"],
                "error": str(e)
            }
    
    def evaluate_batch(self, customer_messages: list[str], 
                       generated_replies: list[str],
                       reference_replies: list[str] = None) -> list[dict]:
        """Evaluate a batch of generated replies."""
        if reference_replies is None:
            reference_replies = [""] * len(customer_messages)
        
        results = []
        for msg, gen, ref in tqdm(
            zip(customer_messages, generated_replies, reference_replies),
            total=len(customer_messages),
            desc="LLM judging"
        ):
            results.append(self.evaluate_single(msg, gen, ref))
        
        return results
    
    def compute_aggregate_scores(self, evaluations: list[dict]) -> dict:
        """
        Compute aggregate statistics from a batch of evaluations.
        """
        # Filter out errors
        valid = [e for e in evaluations if e.get("error") is None]
        
        if not valid:
            return {"error": "No valid evaluations"}
        
        # Per-dimension statistics
        dimension_stats = {}
        for dim in self.DIMENSIONS:
            scores = [e["scores"][dim] for e in valid]
            dimension_stats[dim] = {
                "mean": round(float(np.mean(scores)), 2),
                "median": round(float(np.median(scores)), 2),
                "std": round(float(np.std(scores)), 2),
                "min": int(min(scores)),
                "max": int(max(scores)),
                "distribution": {
                    str(i): scores.count(i) for i in range(1, 6)
                }
            }
        
        # Overall statistics
        overall_scores = [e["overall_score"] for e in valid]
        
        # Collect all major issues
        all_issues = []
        for e in valid:
            all_issues.extend(e.get("major_issues", []))
        
        # Count issue frequency
        from collections import Counter
        issue_counts = Counter(all_issues)
        
        return {
            "avg_overall_score": round(float(np.mean(overall_scores)), 2),
            "median_overall_score": round(float(np.median(overall_scores)), 2),
            "std_overall_score": round(float(np.std(overall_scores)), 2),
            "dimension_stats": dimension_stats,
            "total_evaluated": len(valid),
            "total_errors": len(evaluations) - len(valid),
            "score_distribution": {
                "excellent (4.5-5.0)": sum(1 for s in overall_scores if s >= 4.5),
                "good (3.5-4.5)": sum(1 for s in overall_scores if 3.5 <= s < 4.5),
                "acceptable (2.5-3.5)": sum(1 for s in overall_scores if 2.5 <= s < 3.5),
                "poor (1.5-2.5)": sum(1 for s in overall_scores if 1.5 <= s < 2.5),
                "bad (1.0-1.5)": sum(1 for s in overall_scores if s < 1.5),
            },
            "top_issues": issue_counts.most_common(10)
        }


def compute_judge_human_agreement(judge_scores: list[dict], 
                                   human_scores: list[dict]) -> dict:
    """
    Compute agreement between LLM judge and human scores.
    
    Metrics:
    - Pearson correlation per dimension
    - Cohen's Kappa (discretized to 1-5)
    - Mean absolute error
    """
    from scipy import stats as scipy_stats
    
    agreement = {}
    
    for dim in LLMJudge.DIMENSIONS:
        j_scores = [j["scores"].get(dim, 3) for j in judge_scores]
        h_scores = [h["scores"].get(dim, 3) for h in human_scores]
        
        if len(j_scores) < 2:
            continue
        
        # Pearson correlation
        corr, p_value = scipy_stats.pearsonr(j_scores, h_scores)
        
        # Mean absolute error
        mae = float(np.mean(np.abs(np.array(j_scores) - np.array(h_scores))))
        
        # Exact agreement rate
        exact_match = sum(1 for j, h in zip(j_scores, h_scores) if j == h) / len(j_scores)
        
        # Within-1 agreement (judge score within ±1 of human)
        within_1 = sum(1 for j, h in zip(j_scores, h_scores) if abs(j - h) <= 1) / len(j_scores)
        
        agreement[dim] = {
            "pearson_r": round(float(corr), 3),
            "p_value": round(float(p_value), 4),
            "mae": round(mae, 3),
            "exact_agreement": round(exact_match, 3),
            "within_1_agreement": round(within_1, 3)
        }
    
    # Overall
    j_overall = [j["overall_score"] for j in judge_scores]
    h_overall = [h["overall_score"] for h in human_scores]
    
    if len(j_overall) >= 2:
        corr, p_value = scipy_stats.pearsonr(j_overall, h_overall)
        mae = float(np.mean(np.abs(np.array(j_overall) - np.array(h_overall))))
        
        agreement["overall"] = {
            "pearson_r": round(float(corr), 3),
            "p_value": round(float(p_value), 4),
            "mae": round(mae, 3),
            "num_samples": len(j_overall)
        }
    
    return agreement


def save_judge_agreement(agreement: dict, output_path: Path = None):
    """Save judge-human agreement report."""
    if output_path is None:
        output_path = RESULTS_DIR / "judge_agreement.json"
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(agreement, f, indent=2)
    
    print(f"\n💾 Judge agreement report saved to {output_path}")
    
    # Print summary
    print("\n📊 Judge-Human Agreement:")
    for dim, stats in agreement.items():
        if isinstance(stats, dict) and "pearson_r" in stats:
            print(f"   {dim:20s}: r={stats['pearson_r']:.3f}, MAE={stats['mae']:.3f}, "
                  f"within-1={stats.get('within_1_agreement', 'N/A')}")
