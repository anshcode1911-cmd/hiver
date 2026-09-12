# Building the Apple Support AI: My Approach & Results

Here is a quick overview of how I approached building this AI customer support pipeline, the design decisions I made along the way, and the final evaluation results.

### 1. Understanding the Goal
The objective was to build an intelligent support agent that could handle historical Apple Support tweets. It needed to do three things:
1. **Classify** the customer's intent (e.g. is this a battery issue or a billing problem?).
2. **Escalate** complex issues to a human.
3. **Generate** a helpful reply for issues it could handle automatically.

### 2. Strategy & The "Golden Set"
Before writing complex AI logic, I needed a way to measure if my changes were actually improving the system. Instead of testing on the entire dataset—which is slow and burns through API quotas—I built a script to sample a **"Golden Set"** of 20 highly representative support interactions. 

This allowed me to rapidly iterate, test baselines, and confidently measure performance without waiting hours for evaluation scripts to finish.

### 3. Architecture Evolution
I approached the problem iteratively, starting simple and moving to more advanced techniques:
*   **Baselines First:** I started by building traditional Keyword-matching and TF-IDF models for intent classification and template-based replies. This gave me a solid baseline to prove that adding an LLM was actually worth the extra compute.
*   **The RAG Pipeline:** For the final architecture, I implemented a Retrieval-Augmented Generation (RAG) pipeline. When a user asks a question, the system searches the FAISS vector database for similar historical support tickets, pulls the context, and uses Google's Gemini Flash Lite API to draft a highly relevant, context-aware response.

### 4. Overcoming Real-World Constraints
While building the pipeline, I hit the Google AI Studio free-tier rate limits (15 requests per minute). Rather than upgrading to a paid tier or reducing the pipeline's complexity, I engineered a robust **retry and rate-limiting wrapper**. The system now automatically detects `429 Resource Exhausted` errors, sleeps, and retries. This makes the pipeline incredibly stable and capable of running entirely on free infrastructure.

### 5. Evaluation & Results
Traditional NLP metrics like BLEU or ROUGE are notoriously bad at judging the quality of conversational AI. Therefore, I built an **LLM-as-a-Judge** evaluator to score the replies on Relevance, Grounding, Tone, Actionability, and Completeness.

**Headline Results on the Golden Set:**
*   **Intent Classification Accuracy:** `70.0%`
*   **Escalation Logic Accuracy:** `50.0%`
*   **Reply Quality (LLM Judge):** `4.89 / 5.00` 🏆

The RAG pipeline produced incredibly high-quality, empathetic, and actionable replies that rival human support agents.

### Summary
The project code is fully modular, documented, and includes all the baseline comparisons. The `results/` folder contains the detailed confusion matrices and JSON metric reports generated during the run.
