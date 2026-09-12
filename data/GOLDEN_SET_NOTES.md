# Golden Evaluation Set — Methodology Notes

## Overview
- **Total examples**: 20
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
- `update_installation`: 6 examples
- `app_issue`: 3 examples
- `other`: 2 examples
- `battery_performance`: 2 examples
- `device_troubleshooting`: 2 examples
- `feedback_complaint`: 2 examples
- `billing_payment`: 1 examples
- `connectivity`: 1 examples
- `icloud_storage`: 1 examples

#### By Difficulty
- Classified each message into easy/medium/hard using heuristics:
  - **Easy**: Clear single intent, standard phrasing (>15 chars, no excessive punctuation)
  - **Medium**: Some ambiguity, informal language, one complexity signal
  - **Hard**: Multi-intent, sarcasm, very short, emoji-heavy, CAPS, high emotion
- Targeted mix: 50% easy, 30% medium, 20% hard

**Distribution:**
- `easy`: 18 examples
- `medium`: 2 examples

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
- `auto_handle`: 17 examples
- `escalate`: 3 examples

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
