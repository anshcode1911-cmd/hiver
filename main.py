"""
Main Pipeline Runner — End-to-end pipeline for the Hiver AI Support Agent.

Usage:
    python main.py                    # Run full pipeline
    python main.py --step download    # Run only data download
    python main.py --step preprocess  # Run only preprocessing
    python main.py --step evaluate    # Run only evaluation
    python main.py --step demo       # Run interactive demo
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, RESULTS_DIR, OPENAI_API_KEY


def check_prerequisites():
    """Check that all prerequisites are met."""
    issues = []
    
    if not OPENAI_API_KEY:
        issues.append("❌ OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    
    try:
        import pandas
        import sklearn
        import faiss
        import openai
        import nltk
        import rouge_score
    except ImportError as e:
        issues.append(f"❌ Missing dependency: {e}. Run: pip install -r requirements.txt")
    
    if issues:
        print("\n⚠️  Prerequisites check failed:")
        for issue in issues:
            print(f"   {issue}")
        print("\nFix the issues above and try again.")
        sys.exit(1)
    else:
        print("✅ All prerequisites met")


def step_download():
    """Step 1: Download and filter the dataset."""
    from src.data.download_data import run_download
    print("\n" + "=" * 60)
    print("STEP 1: DATA DOWNLOAD")
    print("=" * 60)
    return run_download()


def step_preprocess():
    """Step 2: Preprocess — reconstruct threads, extract pairs."""
    from src.data.preprocess import run_preprocess
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)
    return run_preprocess()


def step_build_index():
    """Step 3: Build the FAISS retrieval index."""
    from src.reply.retriever import SupportRetriever
    
    print("\n" + "=" * 60)
    print("STEP 3: BUILDING RETRIEVAL INDEX")
    print("=" * 60)
    
    retriever = SupportRetriever()
    
    # Check if index already exists
    if retriever.load():
        print("✅ Index already exists. Skipping rebuild.")
        return retriever
    
    # Load pairs
    pairs_path = DATA_DIR / "apple_pairs.jsonl"
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    
    # Build index
    retriever.build_index(pairs, max_pairs=1000)
    return retriever


def step_create_golden_set():
    """Step 4: Create the golden evaluation set."""
    from src.data.create_golden_set import create_golden_set
    
    print("\n" + "=" * 60)
    print("STEP 4: CREATING GOLDEN EVALUATION SET")
    print("=" * 60)
    
    return create_golden_set()


def step_evaluate():
    """Step 5: Run the full evaluation."""
    from src.evaluation.evaluate import run_full_evaluation
    
    print("\n" + "=" * 60)
    print("STEP 5: FULL EVALUATION")
    print("=" * 60)
    
    return run_full_evaluation()


def step_demo():
    """Interactive demo — process a single customer message through the full pipeline."""
    from src.intents.intent_classifier import LLMClassifier
    from src.reply.reply_generator import RAGReplyGenerator
    from src.reply.retriever import SupportRetriever
    from src.escalation.escalation import HybridEscalation
    
    print("\n" + "=" * 60)
    print("🤖 INTERACTIVE DEMO — @AppleSupport AI Agent")
    print("=" * 60)
    print("Type a customer message and see the agent's response.")
    print("Type 'quit' to exit.\n")
    
    # Initialize components
    classifier = LLMClassifier()
    retriever = SupportRetriever()
    retriever.load()
    generator = RAGReplyGenerator(retriever=retriever)
    escalation = HybridEscalation()
    
    while True:
        try:
            message = input("\n👤 Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not message or message.lower() == "quit":
            break
        
        print("\n   Processing...\n")
        
        # 1. Classify intent
        intent_result = classifier.classify(message)
        print(f"   🏷️  Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
        print(f"   📝 Reasoning: {intent_result['reasoning']}")
        
        # 2. Escalation decision
        esc_result = escalation.decide(message, intent_result["intent"])
        emoji = "🚨" if esc_result["decision"] == "escalate" else "🤖"
        print(f"   {emoji} Decision: {esc_result['decision']} (confidence: {esc_result['confidence']:.2f})")
        print(f"   📝 Reason: {esc_result['reason']}")
        
        # 3. Generate reply
        reply_result = generator.generate(message, intent_result["intent"])
        print(f"\n   💬 @AppleSupport: {reply_result['reply']}")
        print(f"   📊 Grounding: {reply_result['grounding']}")
    
    print("\n👋 Demo ended. Thank you!")


def run_full_pipeline():
    """Run the complete pipeline end-to-end."""
    start_time = time.time()
    
    print("🚀 HIVER AI SUPPORT AGENT — FULL PIPELINE")
    print("=" * 60)
    print(f"Target Brand: @AppleSupport")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    check_prerequisites()
    
    # Step 1: Download data
    step_download()
    
    # Step 2: Preprocess
    step_preprocess()
    
    # Step 3: Build retrieval index
    step_build_index()
    
    # Step 4: Create golden set (if not exists)
    golden_path = DATA_DIR / "golden_set.jsonl"
    if not golden_path.exists():
        step_create_golden_set()
    else:
        print(f"\n✅ Golden set already exists at {golden_path}")
    
    # Step 5: Evaluate
    report = step_evaluate()
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"🏁 PIPELINE COMPLETE")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"   Results: {RESULTS_DIR}")
    print(f"{'=' * 60}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent Pipeline")
    parser.add_argument(
        "--step", 
        choices=["download", "preprocess", "index", "golden", "evaluate", "demo", "all"],
        default="all",
        help="Which pipeline step to run (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.step == "download":
        step_download()
    elif args.step == "preprocess":
        step_preprocess()
    elif args.step == "index":
        step_build_index()
    elif args.step == "golden":
        step_create_golden_set()
    elif args.step == "evaluate":
        step_evaluate()
    elif args.step == "demo":
        step_demo()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
