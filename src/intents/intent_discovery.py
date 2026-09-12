"""
Intent Discovery — Uses LLM to analyze customer messages and validate/refine the intent taxonomy.
This is a one-time analysis script, not part of the production pipeline.
"""
import json
import sys
import random
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import get_openai_client_kwargs, DISCOVERY_MODEL, DATA_DIR, INTENT_TAXONOMY


def discover_intents(pairs: list[dict], sample_size: int = 500) -> dict:
    """
    Use LLM to discover intent categories from a sample of customer messages.
    This helps validate and refine the manually curated taxonomy.
    """
    client = OpenAI(**get_openai_client_kwargs())
    
    # Sample customer messages
    messages = [p["customer_message"] for p in pairs]
    if len(messages) > sample_size:
        messages = random.sample(messages, sample_size)
    
    # Format messages for the prompt
    messages_text = "\n".join(f"{i+1}. {msg}" for i, msg in enumerate(messages[:100]))
    
    prompt = f"""Analyze these 100 customer support messages sent to @AppleSupport on Twitter.
Your task is to identify the most common categories (intents) of customer issues.

Messages:
{messages_text}

Instructions:
1. Identify 10-15 distinct intent categories
2. For each category, provide:
   - A short snake_case name
   - A description
   - 3 example messages from the list above
   - Estimated percentage of messages in this category
3. Categories should be mutually exclusive and collectively exhaustive
4. Include an "other" category for messages that don't fit

Respond in JSON format:
{{
    "intents": [
        {{
            "name": "intent_name",
            "description": "what this category covers",
            "examples": ["msg1", "msg2", "msg3"],
            "estimated_pct": 10
        }}
    ],
    "total_messages_analyzed": 100,
    "notes": "any observations about the data"
}}"""

    print("🔍 Discovering intents from customer messages via LLM...")
    
    response = client.chat.completions.create(
        model=DISCOVERY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=3000
    )
    
    result = json.loads(response.choices[0].message.content)
    
    print(f"\n📊 Discovered {len(result.get('intents', []))} intent categories:")
    for intent in result.get("intents", []):
        print(f"   • {intent['name']}: {intent['description']} (~{intent.get('estimated_pct', '?')}%)")
    
    if result.get("notes"):
        print(f"\n📝 Notes: {result['notes']}")
    
    return result


def validate_taxonomy(discovered: dict) -> dict:
    """
    Compare discovered intents with our pre-defined taxonomy.
    Returns a report of coverage and gaps.
    """
    discovered_names = {i["name"].lower() for i in discovered.get("intents", [])}
    taxonomy_names = {k.lower() for k in INTENT_TAXONOMY.keys()}
    
    report = {
        "taxonomy_intents": sorted(taxonomy_names),
        "discovered_intents": sorted(discovered_names),
        "in_taxonomy_not_discovered": sorted(taxonomy_names - discovered_names),
        "discovered_not_in_taxonomy": sorted(discovered_names - taxonomy_names),
        "overlap": sorted(taxonomy_names & discovered_names),
        "coverage_pct": len(taxonomy_names & discovered_names) / len(taxonomy_names) * 100
    }
    
    print("\n📋 Taxonomy Validation Report:")
    print(f"   Pre-defined intents: {len(taxonomy_names)}")
    print(f"   Discovered intents: {len(discovered_names)}")
    print(f"   Overlap: {len(report['overlap'])}")
    print(f"   Coverage: {report['coverage_pct']:.0f}%")
    
    if report["discovered_not_in_taxonomy"]:
        print(f"   ⚠️  Not in taxonomy: {report['discovered_not_in_taxonomy']}")
    if report["in_taxonomy_not_discovered"]:
        print(f"   ℹ️  Not discovered: {report['in_taxonomy_not_discovered']}")
    
    return report


def run_discovery():
    """Run the full discovery pipeline."""
    pairs_path = DATA_DIR / "apple_pairs.jsonl"
    
    if not pairs_path.exists():
        print("❌ Pairs file not found. Run preprocessing first.")
        return
    
    # Load pairs
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    
    print(f"📄 Loaded {len(pairs):,} pairs")
    
    # Discover intents
    discovered = discover_intents(pairs)
    
    # Validate against taxonomy
    report = validate_taxonomy(discovered)
    
    # Save results
    output_path = DATA_DIR / "intent_discovery_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "discovered": discovered,
            "validation": report
        }, f, indent=2)
    
    print(f"\n💾 Saved discovery results to {output_path}")
    return discovered, report


if __name__ == "__main__":
    run_discovery()
