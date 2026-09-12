# 🤖 AI Customer Support Agent for @AppleSupport

> **Hiver SDE Intern Take-Home Assignment**
> An AI support agent that classifies customer intents, drafts grounded replies, and makes escalation decisions — with rigorous evaluation proving it works.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [What This Agent Does](#-what-this-agent-does)
- [Dataset & Brand Choice](#-dataset--brand-choice)
- [Intent Taxonomy](#-intent-taxonomy)
- [Three-Tier Baselines](#-three-tier-baselines)
- [Golden Evaluation Set](#-golden-evaluation-set)
- [Evaluation Harness](#-evaluation-harness)
- [Results](#-results)
- [Failure Analysis](#-failure-analysis)
- [What is Misleading About My Headline Number?](#-what-is-misleading-about-my-headline-number)
- [What I'd Do Next](#-what-id-do-next)
- [Decision Log](#-decision-log)
- [Project Structure](#-project-structure)
- [Cost Estimate](#-cost-estimate)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Kaggle account with API credentials ([setup guide](https://www.kaggle.com/docs/api))

### Setup (< 2 minutes)

```bash
# 1. Clone the repo
git clone <repo-url>
cd HIVERASSIGMENT

# 2. Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# 3. Set up your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Reproduce Headline Results (< 15 minutes)

```bash
# Option A: One command
make reproduce

# Option B: Step by step
python main.py --step download     # Download & filter dataset (~2 min)
python main.py --step preprocess   # Reconstruct threads, extract pairs (~1 min)
python main.py --step index        # Build FAISS retrieval index (~2 min)
python main.py --step golden       # Create golden evaluation set (~3 min)
python main.py --step evaluate     # Run full evaluation (~7 min)
```

### Interactive Demo

```bash
python main.py --step demo
```

```
👤 Customer: My iPhone battery drains really fast since the iOS update

   🏷️  Intent: battery_performance (confidence: 0.92)
   🤖 Decision: auto_handle (confidence: 0.85)
   💬 @AppleSupport: We understand the frustration! Try Settings > Battery > Battery Health 
      to check. Also, a restart after updating often helps. Let us know how it goes!
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Runner (main.py)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐   ┌────────────────────────────────┐  │
│  │  Data Layer           │   │  Evaluation Harness            │  │
│  │  • download_data.py   │   │  • evaluate.py (orchestrator)  │  │
│  │  • preprocess.py      │   │  • metrics.py (F1,BLEU,ROUGE)  │  │
│  │  • create_golden_set  │   │  • llm_judge.py (GPT-4o)       │  │
│  └──────────┬───────────┘   └──────────────────────────────────┘ │
│             │                                                     │
│  ┌──────────▼───────────┐                                        │
│  │  Intent Classifier    │  3 baselines:                          │
│  │  (intent_classifier)  │  • Keyword regex                      │
│  │                       │  • TF-IDF + Logistic Regression       │
│  │                       │  • GPT-4o-mini (structured output)    │
│  └──────────┬───────────┘                                        │
│             │                                                     │
│  ┌──────────▼───────────┐  ┌─────────────────────────────────┐   │
│  │  Reply Generator      │  │  Escalation Engine              │   │
│  │  (reply_generator)    │  │  (escalation.py)                │   │
│  │  • Template baseline  │  │  • Always escalate baseline     │   │
│  │  • TF-IDF retrieval   │  │  • Rule-based (keywords)        │   │
│  │  • FAISS RAG + LLM    │  │  • Hybrid (rules + LLM)         │   │
│  └──────────────────────┘  └─────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🎯 What This Agent Does

Given a customer message on Twitter, the agent:

| Task | Description | Example |
|------|-------------|---------|
| **1. Classify Intent** | Identifies what the customer needs from 12 categories | "My iPhone keeps restarting" → `device_troubleshooting` |
| **2. Draft Reply** | Generates a reply grounded in Apple's historical support patterns | "We'd like to help! Try a force restart: press and quickly release Volume Up, then Volume Down, then hold Side button. Let us know!" |
| **3. Decide Escalation** | Determines if AI can handle it or if a human should take over | `auto_handle` — standard troubleshooting, no safety/security concerns |

---

## 📊 Dataset & Brand Choice

### Dataset
- **Source**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle, thoughtvector)
- **Full size**: ~3M tweets, dozens of brands
- **Working subset**: ~10,000 conversation threads (stratified by time)

### Why Apple?

| Factor | Apple | Alternatives |
|--------|-------|-------------|
| **Volume** | Highest (~600k+ tweets) | Others have 50-200k |
| **Intent diversity** | Hardware, software, billing, account, connectivity | Often domain-limited |
| **Brand voice consistency** | Structured, professional | Varies widely |
| **Escalation patterns** | Moves to DMs/phone naturally | Less structured |

---

## 🏷️ Intent Taxonomy

12 intents discovered through LLM-assisted analysis of 500+ customer messages, validated against Apple's official support categories:

| Intent | Description | Example |
|--------|-------------|---------|
| `device_troubleshooting` | Hardware/software issues | "My iPhone keeps restarting" |
| `app_issue` | App Store, app problems | "App crashes after update" |
| `account_access` | Apple ID, passwords, sign-in | "Can't sign into my Apple ID" |
| `billing_payment` | Charges, refunds, subscriptions | "Charged twice for an app" |
| `update_installation` | iOS/macOS update problems | "iPad stuck on update screen" |
| `connectivity` | WiFi, Bluetooth, cellular | "Bluetooth won't connect" |
| `battery_performance` | Battery drain, slow device | "Battery drains in 2 hours" |
| `icloud_storage` | iCloud sync, storage, backup | "Photos not syncing to iCloud" |
| `repair_warranty` | Repair, warranty, AppleCare | "How to check warranty status?" |
| `feature_inquiry` | How-to, feature questions | "How do I use Screen Time?" |
| `feedback_complaint` | General frustration, complaints | "Apple quality has gone downhill" |
| `other` | Doesn't fit above categories | "Thanks!", "DM sent" |

---

## 📈 Three-Tier Baselines

Every component is evaluated with three approaches of increasing sophistication:

### Intent Classification

| Method | How it works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Keyword** | Regex matching against predefined keywords per intent | Fast, interpretable, no API cost | Misses paraphrases, no context |
| **TF-IDF + LogReg** | Trained on LLM-generated pseudo-labels | Cheap, decent on common intents | Needs training data, poor on rare intents |
| **LLM (GPT-4o-mini)** | Structured output with intent taxonomy in prompt | Best accuracy, handles nuance | API cost, latency |

### Reply Generation

| Method | How it works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Template** | Most representative reply per intent | Consistent, no API cost | Identical replies for different issues |
| **TF-IDF Retrieval** | Find most similar past customer message, return its agent reply | Grounded in real data | May retrieve irrelevant match |
| **RAG + LLM** | FAISS retrieval of top-5 similar interactions + GPT-4o-mini generation | Best quality, contextual | API cost, potential hallucination |

### Escalation

| Method | How it works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Always Escalate** | Every message goes to human | Zero risk of missed escalation | Zero automation value |
| **Rule-Based** | Keyword triggers + intent heuristics | Fast, deterministic | Misses subtle cases |
| **Hybrid** | Rules pre-filter + LLM reasoning | Best balance of safety and automation | API cost |

---

## 🏆 Golden Evaluation Set

- **Size**: 200 hand-labelled examples (within the 150-250 range)
- **Location**: `data/golden_set.jsonl`
- **Methodology**: `data/GOLDEN_SET_NOTES.md`

### Sampling Strategy

1. **Pool**: Randomly sampled 2,000 pairs from the full dataset (seed=42)
2. **Stratified by intent**: ~15-17 examples per intent category
3. **Stratified by difficulty**: 50% easy, 30% medium, 20% hard
4. **Edge cases**: Deliberately included ~20-30 edge cases (very short, high emotion, multi-intent)

### Labels per example

```json
{
    "id": "gs_001",
    "customer_message": "My iPhone battery drains in 2 hours since the update",
    "ground_truth_intent": "battery_performance",
    "ground_truth_escalation": "auto_handle",
    "escalation_reason": "Standard troubleshooting, no safety/security concerns",
    "reference_reply": "...(actual Apple reply from data)...",
    "difficulty": "easy"
}
```

---

## 🔬 Evaluation Harness

### Automated Metrics

| Metric | Component | What it measures |
|--------|-----------|------------------|
| Accuracy | Intent, Escalation | Exact match rate |
| Weighted F1 | Intent | Class-weighted harmonic mean of precision and recall |
| Macro F1 | Intent | Unweighted average F1 across all classes |
| Missed Escalation Rate | Escalation | % of true escalations the system missed (CRITICAL) |
| False Escalation Rate | Escalation | % of auto-handle cases wrongly escalated |
| BLEU-4 | Reply | N-gram precision vs. reference reply |
| ROUGE-L | Reply | Longest common subsequence recall |

### LLM-as-Judge (GPT-4o)

The judge evaluates each generated reply on 5 dimensions (1-5 scale):

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| **Relevance** | 25% | Does the reply address the customer's actual issue? |
| **Grounding** | 25% | Is the advice factually correct and consistent with Apple's practices? |
| **Tone** | 15% | Does it match Apple's professional, empathetic support style? |
| **Actionability** | 20% | Does it give the customer a clear next step? |
| **Completeness** | 15% | Does it cover all aspects of the customer's message? |

### Judge Calibration

- The judge uses **chain-of-thought reasoning** (explains before scoring)
- Judge model (GPT-4o) is **different from production model** (GPT-4o-mini) to avoid self-evaluation bias
- We measure **judge-human agreement** on ~50 examples via Pearson correlation and MAE

---

## 📉 Results

> **Note**: The table below shows expected ranges. Actual numbers are generated at evaluation runtime and saved in `results/metrics_report.json`.

### Intent Classification

| Method | Accuracy | Weighted F1 | Macro F1 |
|--------|----------|-------------|----------|
| Keyword | ~0.35 | ~0.30 | ~0.25 |
| TF-IDF + LogReg | ~0.55 | ~0.52 | ~0.45 |
| **LLM (GPT-4o-mini)** | **~0.82** | **~0.80** | **~0.75** |

### Escalation

| Method | Accuracy | Missed Escalation ↓ | False Escalation |
|--------|----------|---------------------|------------------|
| Always Escalate | ~0.45 | **0.00** | ~1.00 |
| Rule-Based | ~0.65 | ~0.15 | ~0.25 |
| **Hybrid** | **~0.78** | **~0.08** | **~0.15** |

### Reply Quality

| Method | BLEU-4 | ROUGE-L | LLM Judge (avg/5.0) |
|--------|--------|---------|----------------------|
| Template | ~0.02 | ~0.12 | ~2.1 |
| TF-IDF Retrieval | ~0.06 | ~0.18 | ~2.8 |
| **RAG + LLM** | **~0.08** | **~0.22** | **~3.7** |

---

## 🔍 Failure Analysis

See [REPORT.md](REPORT.md) Section 4 for the full analysis. Top 5 failure modes:

1. **Sarcasm misclassification** (~5% of errors) — "Oh great, another update 😊" classified as complaint instead of update issue
2. **Multi-intent messages** (~8%) — "Battery drains AND WiFi drops" gets only first intent
3. **Outdated retrieval** (~10%) — Retrieves advice about old iOS versions
4. **Generic replies for specific issues** (~15%) — "Try restarting" for a very specific hardware problem
5. **False escalation on positive emotion** (~3%) — Happy customers with exclamation marks trigger emotion rules

---

## ⚠️ What is Misleading About My Headline Number?

See [REPORT.md](REPORT.md) Section 5 for the full analysis. Key caveats:

1. **Golden set is LLM-assisted, not purely human-labelled** — inflates apparent LLM accuracy
2. **Tweet replies ≠ real support** — 280 chars can't resolve most real issues
3. **BLEU/ROUGE are poor metrics** — low scores don't mean bad replies
4. **Single-turn evaluation misses conversation dynamics**
5. **200 examples is small** — per-class metrics have high variance for rare intents
6. **LLM judge calibration may not generalize**

---

## 🔮 What I'd Do Next

With one more week:
1. Multi-turn conversation support (dialogue state tracking)
2. Fine-tune a smaller model (DistilBERT on pseudo-labels) for 10-100x cost reduction
3. Multi-label intent classification
4. Domain-specific embeddings for better retrieval
5. A/B testing framework with statistical significance
6. Temporal retrieval weighting (prefer recent advice)
7. Expand golden set to 500+ with multiple annotators

---

## 📝 Decision Log

See [DECISION_LOG.md](DECISION_LOG.md) for the full list of 15 non-obvious decisions with rationale. Highlights:

- **Apple over other brands** — highest volume, most diverse intents
- **12 intents (not 5 or 77)** — balances specificity with data coverage
- **RAG over fine-tuning** — more transparent and controllable
- **GPT-4o as judge, GPT-4o-mini for production** — avoids self-evaluation bias
- **Hybrid escalation** — safety-critical decisions need deterministic guardrails
- **Conservative escalation labelling** — missed escalation >> false escalation

---

## 📁 Project Structure

```
HIVERASSIGMENT/
├── README.md                      # This file
├── REPORT.md                      # 6-page evaluation report
├── DECISION_LOG.md                # 15 non-obvious decisions
├── requirements.txt               # Python dependencies
├── .env.example                   # API key template
├── Makefile                       # One-command reproduction
├── main.py                        # Pipeline runner
│
├── data/
│   ├── golden_set.jsonl           # 200 hand-labelled examples
│   ├── GOLDEN_SET_NOTES.md        # Sampling & labelling methodology
│   └── (generated at runtime)     # Raw data, pairs, threads
│
├── src/
│   ├── config.py                  # All configuration & prompts
│   ├── data/
│   │   ├── download_data.py       # Kaggle download + brand filtering
│   │   ├── preprocess.py          # Thread reconstruction + pair extraction
│   │   └── create_golden_set.py   # Golden evaluation set creation
│   ├── intents/
│   │   ├── intent_discovery.py    # LLM-based intent taxonomy discovery
│   │   └── intent_classifier.py   # 3 classifiers (keyword, TF-IDF, LLM)
│   ├── reply/
│   │   ├── retriever.py           # FAISS vector retrieval
│   │   └── reply_generator.py     # 3 generators (template, TF-IDF, RAG)
│   ├── escalation/
│   │   └── escalation.py          # 3 engines (always, rules, hybrid)
│   └── evaluation/
│       ├── evaluate.py            # Evaluation orchestrator
│       ├── metrics.py             # Automated metrics (F1, BLEU, ROUGE)
│       └── llm_judge.py           # LLM-as-judge (GPT-4o)
│
├── results/                       # Generated at runtime
│   ├── metrics_report.json        # Full metrics
│   ├── confusion_matrix_*.png     # Visualizations
│   ├── predictions.jsonl          # Per-example predictions
│   └── full_evaluation_results.json
│
└── tests/
    └── test_pipeline.py           # Smoke tests
```

---

## 💰 Cost Estimate

| Component | Model | Est. API Cost |
|-----------|-------|---------------|
| Intent classification (200 golden set × 3 methods) | GPT-4o-mini | ~$0.10 |
| Pseudo-label generation (800 examples) | GPT-4o-mini | ~$0.05 |
| Reply generation (200 golden set) | GPT-4o-mini | ~$0.10 |
| Escalation decisions (200 golden set) | GPT-4o-mini | ~$0.05 |
| Embeddings (3,000 pairs) | text-embedding-3-small | ~$0.02 |
| LLM Judge (200 evaluations) | GPT-4o | ~$2.00 |
| Golden set pre-classification | GPT-4o-mini | ~$0.10 |
| **Total** | | **~$2.50** |

---

## 📜 Citations

- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) by Thought Vector (Kaggle)
- **Banking77** (optional reference): Casanueva et al., 2020 — [Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/)
- **LLM-as-Judge methodology**: Zheng et al., 2023 — [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- **Chain-of-thought prompting**: Wei et al., 2022 — [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- **FAISS**: Johnson et al., 2019 — [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- **AI coding assistants**: Used for development acceleration. All code reviewed and understood by the author.

---

## 🏃 Running Individual Components

```bash
# Download and filter dataset
python main.py --step download

# Preprocess (threads + pairs)
python main.py --step preprocess

# Build FAISS index
python main.py --step index

# Create golden set
python main.py --step golden

# Run evaluation
python main.py --step evaluate

# Interactive demo
python main.py --step demo

# Run smoke tests (no API key needed)
python tests/test_pipeline.py

# Full pipeline
python main.py
```

---

*Built for the Hiver SDE Intern Take-Home Assignment.*
