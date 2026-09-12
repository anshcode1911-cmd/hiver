.PHONY: setup download preprocess index golden evaluate demo test all clean reproduce

# ── Setup ─────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Individual Steps ──────────────────────────────────────────
download:
	python main.py --step download

preprocess:
	python main.py --step preprocess

index:
	python main.py --step index

golden:
	python main.py --step golden

evaluate:
	python main.py --step evaluate

demo:
	python main.py --step demo

# ── Full Pipeline ─────────────────────────────────────────────
all:
	python main.py

# ── Quick Reproduce (headline results in <15 min) ────────────
reproduce: setup all

# ── Tests ─────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

# ── Clean ─────────────────────────────────────────────────────
clean:
	rm -rf data/apple_support_raw.csv data/apple_pairs.jsonl data/apple_threads.jsonl
	rm -rf data/embeddings/ results/
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
