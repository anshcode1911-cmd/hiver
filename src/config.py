"""
Central configuration for the Hiver AI Support Agent.
All constants, paths, and prompt templates live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DATA_DIR = DATA_DIR / "raw"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Ensure directories exist
for d in [DATA_DIR, RESULTS_DIR, RAW_DATA_DIR, EMBEDDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # For OpenAI-compatible proxies (e.g., AIML API)

# ── Model Configuration ──────────────────────────────────────────────────────
CLASSIFIER_MODEL = "gemini-flash-lite-latest"       # Fast classification
REPLY_MODEL = "gemini-flash-lite-latest"            # For reply generation (fast, free tier)
ESCALATION_MODEL = "gemini-flash-lite-latest"       # For escalation decisions (fast, free tier)
JUDGE_MODEL = "gemini-flash-lite-latest"            # For LLM-as-judge
EMBEDDING_MODEL = "gemini-embedding-001"   # Google's embedding model (3072 dim)
DISCOVERY_MODEL = "gemini-3.6-flash"        # For intent discovery


def get_openai_client_kwargs() -> dict:
    """Return kwargs for OpenAI client, including base_url if configured."""
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return kwargs

# ── Brand Configuration ───────────────────────────────────────────────────────
TARGET_BRAND = "AppleSupport"
TARGET_BRAND_ALIASES = ["AppleSupport", "Apple", "applesupport"]

# ── Subsample Size ────────────────────────────────────────────────────────────
MAX_THREADS = 10000          # Max conversation threads to keep
GOLDEN_SET_SIZE = 5        # Number of hand-labelled examples
RETRIEVER_TOP_K = 5          # Number of similar examples for RAG
MAX_REPLY_LENGTH = 280       # Twitter character limit

# ── Intent Taxonomy ───────────────────────────────────────────────────────────
INTENT_TAXONOMY = {
    "device_troubleshooting": {
        "description": "Hardware or software issues on iPhone, iPad, Mac, Apple Watch, or Apple TV",
        "keywords": ["restart", "frozen", "crash", "won't turn on", "screen", "broken",
                     "not working", "glitch", "bug", "error", "issue", "problem",
                     "malfunction", "defective", "stuck", "unresponsive"],
        "examples": [
            "My iPhone keeps restarting randomly",
            "My MacBook screen is flickering",
            "iPad Pro won't turn on after charging"
        ]
    },
    "app_issue": {
        "description": "Problems with App Store, specific apps, or app functionality",
        "keywords": ["app", "download", "install", "app store", "update app",
                     "app crash", "can't open", "app not working", "app freeze"],
        "examples": [
            "I can't download apps from the App Store",
            "An app keeps crashing after the latest update",
            "App Store says 'cannot connect'"
        ]
    },
    "account_access": {
        "description": "Apple ID, password resets, sign-in issues, two-factor authentication",
        "keywords": ["apple id", "password", "sign in", "log in", "locked out",
                     "two factor", "2fa", "verification", "account", "forgot password",
                     "disabled", "recovery"],
        "examples": [
            "I can't sign into my Apple ID",
            "My account has been disabled",
            "I forgot my Apple ID password and can't reset it"
        ]
    },
    "billing_payment": {
        "description": "Charges, refunds, subscriptions, payment methods, unauthorized purchases",
        "keywords": ["charge", "refund", "bill", "payment", "subscription", "purchase",
                     "money", "charged", "receipt", "invoice", "cancel subscription",
                     "unauthorized", "fraud"],
        "examples": [
            "I was charged twice for an app purchase",
            "How do I get a refund for an accidental purchase?",
            "I see unauthorized charges on my account"
        ]
    },
    "update_installation": {
        "description": "iOS, macOS, watchOS, or tvOS update problems — stuck updates, failed installs",
        "keywords": ["update", "upgrade", "ios", "macos", "software update",
                     "stuck on update", "install", "downloading", "update failed",
                     "won't update"],
        "examples": [
            "My iPad is stuck on the update screen",
            "iOS update failed and now my phone won't boot",
            "I can't install the latest macOS update"
        ]
    },
    "connectivity": {
        "description": "WiFi, Bluetooth, cellular, AirDrop, hotspot connection problems",
        "keywords": ["wifi", "bluetooth", "cellular", "network", "connect",
                     "airdrop", "hotspot", "signal", "internet", "disconnects",
                     "no service", "can't connect"],
        "examples": [
            "Bluetooth won't connect to my car",
            "WiFi keeps dropping on my MacBook",
            "AirDrop is not finding nearby devices"
        ]
    },
    "battery_performance": {
        "description": "Battery drain, overheating, slow performance, storage full",
        "keywords": ["battery", "drain", "charge", "slow", "hot", "overheat",
                     "performance", "laggy", "storage", "memory", "fast drain"],
        "examples": [
            "My iPhone battery drains in 2 hours",
            "Phone gets really hot when charging",
            "My iPad has become very slow after the update"
        ]
    },
    "icloud_storage": {
        "description": "iCloud sync, storage management, backup, Photos, iCloud Drive",
        "keywords": ["icloud", "storage", "backup", "sync", "photos",
                     "icloud drive", "space", "full", "not syncing", "restore"],
        "examples": [
            "My photos aren't syncing to iCloud",
            "iCloud storage is full, how do I manage it?",
            "I can't restore my backup from iCloud"
        ]
    },
    "repair_warranty": {
        "description": "Repair status, warranty coverage, AppleCare, Genius Bar appointments",
        "keywords": ["repair", "warranty", "applecare", "genius bar", "fix",
                     "replacement", "service", "broken screen", "water damage",
                     "cracked"],
        "examples": [
            "How do I check my warranty status?",
            "I need to schedule a Genius Bar appointment",
            "Is my cracked screen covered by AppleCare?"
        ]
    },
    "feature_inquiry": {
        "description": "How-to questions, feature explanations, tips and tricks",
        "keywords": ["how to", "how do i", "can i", "what is", "where is",
                     "feature", "setting", "enable", "disable", "turn on",
                     "turn off", "use", "find"],
        "examples": [
            "How do I use Screen Time?",
            "Where can I find my AirPods settings?",
            "Can I use my Apple Watch without an iPhone?"
        ]
    },
    "feedback_complaint": {
        "description": "General frustration, complaints about Apple, negative sentiment without specific issue",
        "keywords": ["terrible", "worst", "hate", "disappointed", "awful",
                     "garbage", "unacceptable", "ridiculous", "angry", "frustrated",
                     "never again", "scam"],
        "examples": [
            "Apple quality has gone downhill",
            "This is the worst customer service I've ever experienced",
            "I'm so frustrated with my Apple products"
        ]
    },
    "other": {
        "description": "Messages that don't fit any above category, or are too ambiguous to classify",
        "keywords": [],
        "examples": [
            "Thanks!",
            "DM sent",
            "Hello"
        ]
    }
}

INTENT_LIST = list(INTENT_TAXONOMY.keys())

# ── Escalation Rules ─────────────────────────────────────────────────────────
ESCALATION_KEYWORDS = [
    "lawyer", "legal", "sue", "attorney", "lawsuit",
    "stolen", "hacked", "compromised", "unauthorized access",
    "refund", "charged", "money back",
    "physical damage", "water damage", "cracked screen",
    "data loss", "lost everything", "privacy",
    "danger", "safety", "fire", "burning", "smoke",
    "threat", "report", "fcc", "bbb", "complaint"
]

# ── Prompt Templates ─────────────────────────────────────────────────────────

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for @AppleSupport customer service messages on Twitter.

Given a customer message, classify it into exactly ONE of the following intents:

{intent_descriptions}

Respond in JSON format:
{{
    "intent": "<intent_name>",
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}

Rules:
- Choose the MOST SPECIFIC intent that applies.
- If a message contains multiple issues, choose the PRIMARY one.
- Use "other" only if no category fits at all.
- Confidence should reflect how clearly the message maps to the intent.

Customer message: "{message}"
"""

REPLY_GENERATION_PROMPT = """You are @AppleSupport on Twitter. Generate a helpful, professional reply to the customer message below.

BRAND VOICE GUIDELINES:
- Concise and clear (Twitter has a 280 character limit)
- Empathetic but professional — acknowledge the issue first
- Provide specific, actionable next steps
- Use the customer's name if available (from @mention)
- Reference Apple's actual support resources when relevant
- Never make promises about refunds, replacements, or specific outcomes
- If the issue needs more info, ask a specific diagnostic question

CONTEXT:
Customer intent: {intent}
Customer message: "{message}"

Here are similar past interactions from @AppleSupport for reference:
{retrieved_examples}

Generate a reply that:
1. Acknowledges the customer's issue
2. Provides helpful guidance grounded in the examples above
3. Stays under 280 characters
4. Matches Apple's professional support tone

Reply:"""

ESCALATION_PROMPT = """You are a triage system for @AppleSupport. Determine whether the following customer message should be:
- **auto_handle**: The AI agent can respond with standard troubleshooting advice
- **escalate**: A human agent should handle this for safety, complexity, or sensitivity reasons

ESCALATION CRITERIA (escalate if ANY apply):
1. Account security — compromised accounts, unauthorized access, hacking
2. Billing disputes — unauthorized charges, refund requests involving money
3. Hardware failure — device physically damaged, not turning on, safety concerns
4. High emotion — extreme frustration, threats, profanity indicating the customer needs human empathy
5. Multi-step diagnosis — issues requiring back-and-forth troubleshooting that can't be resolved in one tweet
6. Legal/privacy — mentions of lawyers, legal action, data privacy requests
7. Safety concerns — overheating, fire, physical danger

Customer message: "{message}"
Classified intent: {intent}

Respond in JSON format:
{{
    "decision": "auto_handle" or "escalate",
    "reason": "<brief explanation of why>",
    "confidence": <0.0 to 1.0>,
    "criteria_triggered": ["<which criteria above triggered escalation, if any>"]
}}
"""

LLM_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of AI-generated customer support replies for @AppleSupport on Twitter.

EVALUATION CRITERIA (score each 1-5):

1. **Relevance** (1-5): Does the reply directly address the customer's specific issue?
   - 1: Completely off-topic
   - 3: Partially relevant but misses key aspects
   - 5: Perfectly addresses the exact issue raised

2. **Grounding** (1-5): Is the advice factually correct and consistent with Apple's actual support practices?
   - 1: Contains hallucinated or incorrect information
   - 3: Generally correct but some inaccuracies
   - 5: Fully grounded in Apple's real support practices

3. **Tone** (1-5): Does it match Apple's professional, empathetic support style?
   - 1: Rude, dismissive, or inappropriately casual
   - 3: Acceptable but generic
   - 5: Perfectly matches Apple's warm, professional voice

4. **Actionability** (1-5): Does the reply give the customer a clear next step?
   - 1: No actionable guidance at all
   - 3: Vague suggestions
   - 5: Clear, specific next steps the customer can take immediately

5. **Completeness** (1-5): Does it cover all aspects of the customer's message?
   - 1: Addresses less than half the concerns
   - 3: Covers the main issue but misses secondary points
   - 5: Comprehensively addresses everything raised

CUSTOMER MESSAGE: "{customer_message}"
GENERATED REPLY: "{generated_reply}"
REFERENCE REPLY (actual Apple response): "{reference_reply}"

Think step-by-step. First analyze each criterion, then provide scores.

Respond in JSON format:
{{
    "reasoning": {{
        "relevance": "<analysis>",
        "grounding": "<analysis>",
        "tone": "<analysis>",
        "actionability": "<analysis>",
        "completeness": "<analysis>"
    }},
    "scores": {{
        "relevance": <1-5>,
        "grounding": <1-5>,
        "tone": <1-5>,
        "actionability": <1-5>,
        "completeness": <1-5>
    }},
    "overall_score": <1-5 weighted average>,
    "major_issues": ["<list any critical problems>"]
}}
"""


def get_intent_descriptions_for_prompt() -> str:
    """Format intent taxonomy for use in prompts."""
    lines = []
    for intent_name, info in INTENT_TAXONOMY.items():
        examples_str = "; ".join(f'"{e}"' for e in info["examples"])
        lines.append(f"- **{intent_name}**: {info['description']}")
        lines.append(f"  Examples: {examples_str}")
    return "\n".join(lines)
