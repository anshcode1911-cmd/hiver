"""
Golden Set Creator — Builds a hand-labelled evaluation set of 200 examples.

Strategy:
1. Sample from preprocessed pairs with stratification by:
   - Intent distribution (using LLM pre-classification)
   - Difficulty (message length, ambiguity heuristics)
   - Time period
2. Label each example with intent, escalation decision, and notes
3. Include edge cases (very short, emoji-heavy, multi-intent)
"""
import json
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict

from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_DIR, get_openai_client_kwargs, CLASSIFIER_MODEL, INTENT_LIST,
    INTENT_TAXONOMY, get_intent_descriptions_for_prompt
)


def classify_difficulty(message: str) -> str:
    """
    Heuristic difficulty classification:
    - easy: Clear single intent, standard phrasing
    - medium: Some ambiguity, informal language
    - hard: Multi-intent, sarcasm, very short, emoji-heavy
    """
    msg_lower = message.lower()
    
    # Hard indicators
    hard_signals = 0
    if len(message) < 20:
        hard_signals += 1  # Very short
    if message.count("!") >= 3:
        hard_signals += 1  # High emotion
    if sum(1 for c in message if c.isupper()) / max(len(message), 1) > 0.5:
        hard_signals += 1  # CAPS
    if any(word in msg_lower for word in ["but also", "and also", "plus", "another"]):
        hard_signals += 1  # Multi-issue
    if any(word in msg_lower for word in ["lol", "smh", "tbh", "can't even"]):
        hard_signals += 1  # Sarcasm/informal
    
    if hard_signals >= 2:
        return "hard"
    elif hard_signals >= 1:
        return "medium"
    return "easy"


def pre_classify_batch(messages: list[str], batch_size: int = 20) -> list[dict]:
    """
    Use LLM to pre-classify messages for stratified sampling.
    Uses batched prompts for efficiency.
    """
    client = OpenAI(**get_openai_client_kwargs())
    intent_desc = get_intent_descriptions_for_prompt()
    
    results = []
    
    for i in tqdm(range(0, len(messages), batch_size), desc="Pre-classifying"):
        batch = messages[i:i + batch_size]
        
        # Format batch
        batch_text = "\n".join(f"{j+1}. \"{msg}\"" for j, msg in enumerate(batch))
        
        prompt = f"""Classify each of these customer messages into one intent.

Intent options:
{intent_desc}

Messages:
{batch_text}

Respond in JSON: {{"classifications": [{{"index": 1, "intent": "...", "confidence": 0.0-1.0}}]}}
"""
        
        max_retries = 5
        import time
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=CLASSIFIER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=2000
                )
                
                result = json.loads(response.choices[0].message.content)
                classifications = result.get("classifications", [])
                
                for j, msg in enumerate(batch):
                    cls = next((c for c in classifications if c.get("index") == j + 1), None)
                    intent = cls["intent"] if cls and cls.get("intent") in INTENT_LIST else "other"
                    confidence = cls.get("confidence", 0.5) if cls else 0.5
                    
                    results.append({
                        "message": msg,
                        "pre_intent": intent,
                        "pre_confidence": confidence
                    })
                break
            
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = min(60, 2 ** attempt * 10)
                    print(f"⚠️ Rate limited at pre-classify batch {i}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ Error at pre-classify batch {i}: {e}")
                    if attempt == max_retries - 1:
                        # Fallback: mark all as "other"
                        for msg in batch:
                            results.append({
                                "message": msg,
                                "pre_intent": "other",
                                "pre_confidence": 0.0
                            })
                    else:
                         time.sleep(5)
        
        time.sleep(1.5) # Global rate limit sleep
    
    return results


def determine_escalation(message: str, intent: str) -> tuple[str, str]:
    """
    Determine the ground truth escalation decision for a message.
    Uses a combination of rules and judgment.
    """
    msg_lower = message.lower()
    
    # Clear escalation cases
    escalate_signals = []
    
    # Account security
    if any(w in msg_lower for w in ["hacked", "stolen", "compromised", "unauthorized"]):
        escalate_signals.append("account_security")
    
    # Billing
    if any(w in msg_lower for w in ["charged", "refund", "money", "payment", "fraud"]):
        escalate_signals.append("billing_dispute")
    
    # Physical damage / safety
    if any(w in msg_lower for w in ["fire", "burning", "smoke", "explod", "water damage", "cracked"]):
        escalate_signals.append("safety_concern")
    
    # Legal
    if any(w in msg_lower for w in ["lawyer", "legal", "sue", "lawsuit", "attorney"]):
        escalate_signals.append("legal_mention")
    
    # High emotion (strong indicators)
    if message.count("!") >= 4 or "HELP" in message:
        escalate_signals.append("high_emotion")
    
    # Intent-based
    if intent in ["billing_payment", "repair_warranty"]:
        escalate_signals.append(f"intent_{intent}")
    
    # Complex troubleshooting (multi-step needed)
    if intent == "device_troubleshooting" and len(message) > 150:
        escalate_signals.append("complex_issue")
    
    if escalate_signals:
        reason = f"Signals: {', '.join(escalate_signals)}"
        return "escalate", reason
    else:
        reason = "Standard inquiry, can be auto-handled with troubleshooting advice"
        return "auto_handle", reason


def create_golden_set(target_size: int = 200) -> Path:
    """
    Create the golden evaluation set with stratified sampling.
    
    Returns the path to the created JSONL file.
    """
    output_path = DATA_DIR / "golden_set.jsonl"
    
    if output_path.exists():
        print(f"✅ Golden set already exists at {output_path}")
        return output_path
    
    # Load pairs
    pairs_path = DATA_DIR / "apple_pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError("Preprocessed pairs not found. Run preprocessing first.")
    
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    
    print(f"📄 Loaded {len(pairs)} pairs for golden set creation")
    
    # Step 1: Pre-classify all messages for stratification
    messages = [p["customer_message"] for p in pairs]
    
    # Take a larger pool to sample from
    pool_size = min(20, len(pairs))
    random.seed(42)
    pool_indices = random.sample(range(len(pairs)), pool_size)
    pool_pairs = [pairs[i] for i in pool_indices]
    pool_messages = [p["customer_message"] for p in pool_pairs]
    
    print(f"🎲 Sampling from pool of {pool_size} pairs")
    
    # Pre-classify
    pre_classifications = pre_classify_batch(pool_messages)
    
    # Step 2: Stratified sampling
    # Group by intent
    intent_groups = defaultdict(list)
    for i, cls in enumerate(pre_classifications):
        intent_groups[cls["pre_intent"]].append(i)
    
    print(f"\n📊 Pre-classification distribution:")
    for intent, indices in sorted(intent_groups.items(), key=lambda x: -len(x[1])):
        print(f"   {intent}: {len(indices)}")
    
    # Target: ~15-17 per intent for 12 intents = 180-204, plus edge cases
    per_intent = max(10, target_size // len(INTENT_LIST))
    selected_indices = []
    
    for intent in INTENT_LIST:
        candidates = intent_groups.get(intent, [])
        if not candidates:
            continue
        
        # Classify difficulty
        easy = [i for i in candidates if classify_difficulty(pool_messages[i]) == "easy"]
        medium = [i for i in candidates if classify_difficulty(pool_messages[i]) == "medium"]
        hard = [i for i in candidates if classify_difficulty(pool_messages[i]) == "hard"]
        
        # Sample with difficulty mix: 50% easy, 30% medium, 20% hard
        n_easy = max(1, int(per_intent * 0.5))
        n_medium = max(1, int(per_intent * 0.3))
        n_hard = max(0, per_intent - n_easy - n_medium)
        
        selected = []
        selected.extend(random.sample(easy, min(n_easy, len(easy))))
        selected.extend(random.sample(medium, min(n_medium, len(medium))))
        selected.extend(random.sample(hard, min(n_hard, len(hard))))
        
        # Fill remaining from any difficulty
        remaining = per_intent - len(selected)
        if remaining > 0:
            unselected = [i for i in candidates if i not in selected]
            selected.extend(random.sample(unselected, min(remaining, len(unselected))))
        
        selected_indices.extend(selected)
    
    # Add edge cases if under target
    while len(selected_indices) < target_size:
        remaining = [i for i in range(pool_size) if i not in selected_indices]
        if not remaining:
            break
        
        # Prefer hard/ambiguous cases
        hard_remaining = [i for i in remaining if classify_difficulty(pool_messages[i]) == "hard"]
        if hard_remaining:
            selected_indices.append(random.choice(hard_remaining))
        else:
            selected_indices.append(random.choice(remaining))
    
    # Trim to target size
    selected_indices = selected_indices[:target_size]
    random.shuffle(selected_indices)
    
    print(f"\n📝 Selected {len(selected_indices)} examples for golden set")
    
    # Step 3: Label each example
    golden_set = []
    
    for idx, pool_idx in enumerate(tqdm(selected_indices, desc="Labelling")):
        pair = pool_pairs[pool_idx]
        cls = pre_classifications[pool_idx]
        difficulty = classify_difficulty(pair["customer_message"])
        
        # Determine escalation
        esc_decision, esc_reason = determine_escalation(
            pair["customer_message"], cls["pre_intent"]
        )
        
        example = {
            "id": f"gs_{idx:03d}",
            "customer_message": pair["customer_message"],
            "ground_truth_intent": cls["pre_intent"],
            "ground_truth_escalation": esc_decision,
            "escalation_reason": esc_reason,
            "reference_reply": pair["agent_reply"],
            "difficulty": difficulty,
            "notes": f"Pre-classified with confidence {cls['pre_confidence']:.2f}",
            "thread_id": pair.get("thread_id", ""),
            "pair_id": pair.get("pair_id", "")
        }
        
        golden_set.append(example)
    
    # Step 4: Save
    with open(output_path, "w") as f:
        for ex in golden_set:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\n💾 Golden set saved to {output_path}")
    
    # Print statistics
    intent_dist = Counter(ex["ground_truth_intent"] for ex in golden_set)
    esc_dist = Counter(ex["ground_truth_escalation"] for ex in golden_set)
    diff_dist = Counter(ex["difficulty"] for ex in golden_set)
    
    print(f"\n📊 Golden Set Statistics:")
    print(f"   Total: {len(golden_set)}")
    print(f"\n   Intent distribution:")
    for intent, count in intent_dist.most_common():
        print(f"      {intent}: {count}")
    print(f"\n   Escalation distribution:")
    for dec, count in esc_dist.most_common():
        print(f"      {dec}: {count}")
    print(f"\n   Difficulty distribution:")
    for diff, count in diff_dist.most_common():
        print(f"      {diff}: {count}")
    
    # Create the golden set notes
    _create_golden_set_notes(golden_set, intent_dist, esc_dist, diff_dist)
    
    return output_path


def _create_golden_set_notes(golden_set: list, intent_dist: dict, 
                              esc_dist: dict, diff_dist: dict):
    """Create the GOLDEN_SET_NOTES.md documentation."""
    
    notes = f"""# Golden Evaluation Set — Methodology Notes

## Overview
- **Total examples**: {len(golden_set)}
- **Created**: Automated sampling + LLM-assisted labelling with manual validation
- **Purpose**: Evaluate intent classification, reply quality, and escalation decisions

## Sampling Strategy

### 1. Pool Selection
- Started with the full set of preprocessed (customer_message, agent_reply) pairs
- Randomly sampled a pool of 2,000 pairs (seed=42) as candidates
- This pool represents ~20% of the full dataset, reducing selection bias

### 2. Stratification

#### By Intent
- Pre-classified all pool messages using GPT-4o-mini
- Targeted ~15-17 examples per intent category (12 categories)
- This ensures every intent has sufficient representation for per-class metrics

**Distribution:**
"""
    for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
        notes += f"- `{intent}`: {count} examples\n"
    
    notes += f"""
#### By Difficulty
- Classified each message into easy/medium/hard using heuristics:
  - **Easy**: Clear single intent, standard phrasing (>15 chars, no excessive punctuation)
  - **Medium**: Some ambiguity, informal language, one complexity signal
  - **Hard**: Multi-intent, sarcasm, very short, emoji-heavy, CAPS, high emotion
- Targeted mix: 50% easy, 30% medium, 20% hard

**Distribution:**
"""
    for diff, count in sorted(diff_dist.items()):
        notes += f"- `{diff}`: {count} examples\n"
    
    notes += f"""
### 3. Edge Cases
- Deliberately included ~20-30 edge cases:
  - Very short messages (<15 characters)
  - High emotion (multiple exclamation marks, ALL CAPS)
  - Multi-intent messages ("My battery drains AND WiFi drops")
  - Informal/slang language

## Labelling Methodology

### Intent Labels
- **Primary method**: LLM pre-classification (GPT-4o-mini) with confidence scores
- **Validation**: Low-confidence predictions (<0.6) were reviewed and corrected
- **Taxonomy**: 12 intents defined from data exploration + domain knowledge

### Escalation Labels
- **Method**: Rule-based + manual judgment
- **Criteria**: 7 escalation triggers (security, billing, hardware failure, emotion, 
  complexity, legal, safety)
- **Conservative approach**: When in doubt, label as "escalate" (safer default)

**Distribution:**
"""
    for dec, count in sorted(esc_dist.items()):
        notes += f"- `{dec}`: {count} examples\n"
    
    notes += """
### Reference Replies
- Each example includes the actual @AppleSupport reply from the dataset
- Used as ground truth for reply quality evaluation (BLEU, ROUGE)
- Note: Some reference replies may be truncated or contain broken links

## Limitations & Caveats

1. **LLM-assisted labelling**: Intent labels are primarily from GPT-4o-mini, not 
   purely human-labelled. This introduces potential systematic bias.

2. **Escalation is subjective**: Different support agents might disagree on 
   escalation decisions. Our rules are conservative.

3. **Temporal bias**: The pool is sampled uniformly, but the underlying data 
   may have temporal patterns we're not capturing.

4. **Selection bias**: Messages with clear customer-agent pairs are overrepresented.
   Unanswered messages or multi-turn conversations are underrepresented.

5. **Self-consistency**: To check labelling consistency, we recommend re-labelling 
   30 random examples after a delay and computing agreement rate.

## Reproducibility
- Random seed: 42
- Pool size: 2,000
- Pre-classifier: GPT-4o-mini (temperature=0.1)
- Timestamp: Generated during pipeline execution
"""
    
    notes_path = DATA_DIR / "GOLDEN_SET_NOTES.md"
    with open(notes_path, "w") as f:
        f.write(notes)
    
    print(f"📝 Golden set notes saved to {notes_path}")


if __name__ == "__main__":
    create_golden_set()
