from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# ─── Load FinBERT model (downloads once, cached after) ────────────────────────
MODEL_NAME = "ProsusAI/finbert"
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("Loading FinBERT model... (first time takes ~1 min)")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None  # return all labels
        )
        print("FinBERT loaded!")
    return _pipeline

# ─── Risk keyword list ─────────────────────────────────────────────────────────
RISK_KEYWORDS = [
    "risk", "uncertainty", "decline", "loss", "headwind", "challenge",
    "pressure", "concern", "volatile", "downturn", "slowdown", "miss",
    "below expectations", "supply chain", "inflation", "recession"
]

POSITIVE_KEYWORDS = [
    "growth", "record", "exceed", "strong", "beat", "momentum",
    "expansion", "profitable", "raised guidance", "outperform",
    "robust", "acceleration", "ahead of expectations"
]

GUIDANCE_PHRASES = [
    "we expect", "we anticipate", "guidance", "outlook", "forecast",
    "next quarter", "full year", "going forward", "we project"
]

def chunk_text(text: str, max_tokens: int = 400) -> list:
    """Split text into chunks of ~400 words for FinBERT."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks[:10]  # Max 10 chunks

def analyze_sentiment(text: str) -> dict:
    """
    Run FinBERT on text chunks and return aggregated sentiment analysis.
    Returns: score (-1 to +1), label, confidence, keywords, guidance
    """
    nlp = get_pipeline()
    chunks = chunk_text(text)

    if not chunks:
        return {"score": 0, "label": "neutral", "confidence": 0,
                "risk_keywords": [], "positive_keywords": [], "guidance_signals": []}

    pos_scores, neg_scores, neu_scores = [], [], []

    for chunk in chunks:
        try:
            result = nlp(chunk[:512])[0]  # FinBERT max 512 tokens
            for item in result:
                label = item["label"].lower()
                score = item["score"]
                if label == "positive":
                    pos_scores.append(score)
                elif label == "negative":
                    neg_scores.append(score)
                else:
                    neu_scores.append(score)
        except Exception:
            continue

    avg_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0
    avg_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0
    avg_neu = sum(neu_scores) / len(neu_scores) if neu_scores else 0

    # Composite score: -1 (very negative) to +1 (very positive)
    composite_score = round(avg_pos - avg_neg, 4)

    if composite_score > 0.1:
        label = "positive"
        confidence = round(avg_pos * 100, 1)
    elif composite_score < -0.1:
        label = "negative"
        confidence = round(avg_neg * 100, 1)
    else:
        label = "neutral"
        confidence = round(avg_neu * 100, 1)

    # Extract keywords
    text_lower = text.lower()
    found_risks = [kw for kw in RISK_KEYWORDS if kw in text_lower][:6]
    found_positives = [kw for kw in POSITIVE_KEYWORDS if kw in text_lower][:6]
    found_guidance = [ph for ph in GUIDANCE_PHRASES if ph in text_lower][:4]

    return {
        "score": composite_score,
        "label": label,
        "confidence": confidence,
        "positive_pct": round(avg_pos * 100, 1),
        "negative_pct": round(avg_neg * 100, 1),
        "neutral_pct": round(avg_neu * 100, 1),
        "risk_keywords": found_risks,
        "positive_keywords": found_positives,
        "guidance_signals": found_guidance,
        "chunks_analyzed": len(chunks)
    }
