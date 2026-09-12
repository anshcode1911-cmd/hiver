# Decision Log — Non-Obvious Decisions

This document records the 15 most important non-obvious decisions made during the design and implementation of the AI Support Agent, and the reasoning behind each.

---

## 1. Chose Apple over other brands

**Decision**: Selected `@AppleSupport` as the target brand.

**Alternatives considered**: SpotifyCares (music-specific), AmazonHelp (e-commerce), Delta (airlines).

**Rationale**: Apple has the highest tweet volume (~600k+), the most diverse issue types (hardware, software, billing, account, connectivity), and the most consistent brand voice. This makes both intent classification and reply evaluation more interesting and rigorous. A smaller brand would have been easier but less impressive.

---

## 2. Defined 12 intents (not 5, not 77)

**Decision**: Created a taxonomy of 12 intents: device_troubleshooting, app_issue, account_access, billing_payment, update_installation, connectivity, battery_performance, icloud_storage, repair_warranty, feature_inquiry, feedback_complaint, other.

**Alternatives considered**: 5-6 broad categories (too coarse, loses signal), 30+ fine-grained categories (sparse data per category, harder to evaluate), using Banking77's 77 intents directly (different domain).

**Rationale**: 12 intents balance specificity with having enough examples per class. Each intent maps to a distinct resolution pattern, which is what matters for reply generation. The "other" catch-all prevents forcing ambiguous messages into wrong buckets.

---

## 3. Used LLM-assisted labelling for the golden set (not pure human)

**Decision**: Pre-classified golden set messages using GPT-4o-mini, then validated low-confidence predictions.

**Alternatives considered**: Pure manual labelling (prohibitively time-consuming for 200 examples with 12 categories), using only keyword rules (too noisy).

**Rationale**: LLM-assisted labelling is realistic and acknowledged in the report. The key is transparency — we document this limitation and note that it introduces systematic bias (the same model family evaluates and labels). A truly independent evaluation would use different annotators.

---

## 4. RAG over fine-tuning for reply generation

**Decision**: Used retrieval-augmented generation (FAISS + LLM) rather than fine-tuning a model on Apple's replies.

**Alternatives considered**: Fine-tuning GPT-3.5 on Apple reply pairs, training a smaller model from scratch.

**Rationale**: RAG is more transparent (you can inspect what was retrieved), more controllable (you can filter retrieved examples), and doesn't require a fine-tuning budget. It also generalizes better to novel issues because the retrieval step finds relevant patterns even for unseen intents. Fine-tuning would likely produce better BLEU scores but at the cost of interpretability.

---

## 5. Used GPT-4o as judge, GPT-4o-mini for production

**Decision**: Separated the judge model (GPT-4o) from the production model (GPT-4o-mini).

**Rationale**: Using the same model to generate and evaluate creates self-evaluation bias — the model would rate its own style highly. GPT-4o is a stronger model that can catch subtleties the mini model misses. This separation is a standard practice in LLM evaluation (Zheng et al., 2023).

---

## 6. Hybrid escalation (not pure LLM)

**Decision**: Combined rule-based keyword matching with LLM reasoning for escalation.

**Alternatives considered**: Pure LLM (expensive, slower), pure rules (misses nuanced cases).

**Rationale**: Safety-critical decisions like "should a human handle this?" benefit from deterministic guardrails. The rules catch obvious cases (security keywords, legal mentions) cheaply, while the LLM handles nuanced cases (detecting frustration without explicit keywords, understanding context). This two-stage approach is also faster — most messages are handled by rules alone.

---

## 7. Conservative escalation labelling ("when in doubt, escalate")

**Decision**: Labelled ambiguous golden set examples as "escalate" rather than "auto_handle".

**Rationale**: In production, a missed escalation (AI handles what should go to a human) is far worse than a false escalation (human handles what AI could have done). The asymmetric cost means we should bias toward escalation in ground truth labels. This is explicitly documented as a limitation.

---

## 8. 280-character reply limit enforced

**Decision**: Hard-limited all generated replies to 280 characters (Twitter's limit).

**Alternatives considered**: Allowing multi-tweet replies, ignoring the limit for evaluation purposes.

**Rationale**: This is a real-world constraint that significantly affects reply quality. A good reply within 280 chars is harder than an unconstrained reply but more practically useful. It forces the model to be concise, which matches Apple's actual support style.

---

## 9. Subsample to ~10,000 threads (not full 3M tweets)

**Decision**: Work with a stratified subsample of ~10,000 conversation threads.

**Rationale**: The assignment explicitly says "a subsample is expected and encouraged." 10k threads provide enough data for intent discovery, retrieval grounding, and statistically meaningful evaluation, while keeping the pipeline runnable in <15 minutes. The subsample is time-stratified to avoid temporal bias.

---

## 10. Weighted F1 as primary intent metric (not accuracy)

**Decision**: Report weighted F1 as the primary intent classification metric, not raw accuracy.

**Rationale**: The intent distribution is imbalanced (device_troubleshooting is much more common than repair_warranty). Raw accuracy would be inflated by getting the majority class right. Weighted F1 accounts for class imbalance. We also report macro F1 for completeness (which treats all classes equally).

---

## 11. BLEU + ROUGE + LLM judge (not just one metric)

**Decision**: Used three complementary metrics for reply quality.

**Rationale**: BLEU measures n-gram precision (how much of the generated reply matches the reference), ROUGE measures recall (how much of the reference is captured), and the LLM judge evaluates semantic quality (relevance, tone, actionability). Each metric has known failure modes — BLEU penalizes valid paraphrases, ROUGE rewards extractive copying, and LLM judges have calibration issues. Using all three provides a more complete picture.

---

## 12. Pseudo-labels for TF-IDF training (not golden set leakage)

**Decision**: Trained the TF-IDF classifier on LLM-generated pseudo-labels from a *separate* set of messages, NOT from the golden set.

**Rationale**: Training on golden set labels and evaluating on the same golden set would be data leakage. Instead, we use the LLM to label 800 separate messages as training data. This is honest — the TF-IDF baseline gets a weaker training signal, but the evaluation is fair.

---

## 13. Cosine similarity (normalized inner product) for FAISS

**Decision**: Used normalized vectors with inner product search (equivalent to cosine similarity) instead of L2 distance.

**Rationale**: For text embeddings, cosine similarity is the standard choice because it measures directional similarity regardless of magnitude. L2 distance is affected by vector norms, which can vary based on text length. FAISS IndexFlatIP with normalized vectors gives exact cosine similarity search.

---

## 14. Temperature 0.1 for classification, 0.7 for generation

**Decision**: Used low temperature (0.1) for intent classification and escalation, higher temperature (0.7) for reply generation.

**Rationale**: Classification should be deterministic — the same message should always get the same intent. Low temperature ensures this. Reply generation benefits from some creativity — we want natural-sounding replies, not robotic repetition. 0.7 is a balanced value that allows variation while staying coherent.

---

## 15. Chain-of-thought in judge prompt (not direct scoring)

**Decision**: Required the LLM judge to explain its reasoning before scoring.

**Rationale**: "Think step-by-step" prompting (chain-of-thought) improves the quality and consistency of LLM evaluations (Wei et al., 2022). Without reasoning, judges tend to give inflated scores. By forcing the model to analyze each dimension first, we get more calibrated and defensible scores. The reasoning also helps debug judge failures.
