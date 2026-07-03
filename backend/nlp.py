import requests
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "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")
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

RISK_KEYWORDS = [
    "risk", "uncertainty", "decline", "loss", "headwind", "challenge",
    "pressure", "concern", "volatile", "downturn", "slowdown", "miss",
    "below expectations", "supply chain", "inflation", "recession",
    "weak", "difficult", "disappointing", "negative", "decrease", "fell"
]
POSITIVE_KEYWORDS = [
    "growth", "record", "exceed", "strong", "beat", "momentum",
    "expansion", "profitable", "raised guidance", "outperform",
    "robust", "acceleration", "ahead of expectations", "increase",
    "improved", "higher", "positive", "exceeded", "gained", "rose"
]
GUIDANCE_PHRASES = [
    "we expect", "we anticipate", "guidance", "outlook", "forecast",
    "next quarter", "full year", "going forward", "we project"
]

def chunk_text(text: str, max_words: int = 300) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks[:4]

def query_finbert(text: str) -> list:
    """Call HuggingFace FinBERT API."""
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": text[:512]},
            timeout=30
        )
        if response.status_code == 503:
            # Model loading, wait and retry
            import time
            time.sleep(10)
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={"inputs": text[:512]},
                timeout=30
            )
        return response.json()
    except Exception:
        return []

def analyze_sentiment(text: str) -> dict:
    """Run real FinBERT via HuggingFace Inference API."""
    chunks = chunk_text(text)
    if not chunks:
        return {"score": 0, "label": "neutral", "confidence": 0,
                "risk_keywords": [], "positive_keywords": [], "guidance_signals": []}

    pos_scores, neg_scores, neu_scores = [], [], []

    for chunk in chunks:
        result = query_finbert(chunk)
        if not result or not isinstance(result, list):
            continue
        # Handle nested list response
        items = result[0] if isinstance(result[0], list) else result
        for item in items:
            if not isinstance(item, dict):
                continue
            label = item.get("label", "").lower()
            score = item.get("score", 0)
            if label == "positive":
                pos_scores.append(score)
            elif label == "negative":
                neg_scores.append(score)
            elif label == "neutral":
                neu_scores.append(score)

    # Fallback to keyword if API gave nothing
    if not pos_scores and not neg_scores:
        return keyword_fallback(text)

    avg_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0
    avg_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0
    avg_neu = sum(neu_scores) / len(neu_scores) if neu_scores else 0

    composite = round(avg_pos - avg_neg, 4)

    if composite > 0.1:
        label = "positive"
        confidence = round(avg_pos * 100, 1)
    elif composite < -0.1:
        label = "negative"
        confidence = round(avg_neg * 100, 1)
    else:
        label = "neutral"
        confidence = round(avg_neu * 100, 1)

    text_lower = text.lower()
    return {
        "score": composite,
        "label": label,
        "confidence": confidence,
        "positive_pct": round(avg_pos * 100, 1),
        "negative_pct": round(avg_neg * 100, 1),
        "neutral_pct": round(avg_neu * 100, 1),
        "risk_keywords": [kw for kw in RISK_KEYWORDS if kw in text_lower][:6],
        "positive_keywords": [kw for kw in POSITIVE_KEYWORDS if kw in text_lower][:6],
        "guidance_signals": [ph for ph in GUIDANCE_PHRASES if ph in text_lower][:4],
        "chunks_analyzed": len(chunks)
    }

def keyword_fallback(text: str) -> dict:
    """Fallback keyword scorer if API fails."""
    text_lower = text.lower()
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_hits = sum(1 for kw in RISK_KEYWORDS if kw in text_lower)
    pos_score = min(pos_hits / 10, 1.0)
    neg_score = min(neg_hits / 10, 1.0)
    composite = round(pos_score - neg_score, 4)
    label = "positive" if composite > 0.05 else "negative" if composite < -0.05 else "neutral"
    return {
        "score": composite, "label": label,
        "confidence": round(max(pos_score, neg_score) * 100, 1),
        "positive_pct": round(pos_score * 100, 1),
        "negative_pct": round(neg_score * 100, 1),
        "neutral_pct": round(max(0, 1 - pos_score - neg_score) * 100, 1),
        "risk_keywords": [kw for kw in RISK_KEYWORDS if kw in text_lower][:6],
        "positive_keywords": [kw for kw in POSITIVE_KEYWORDS if kw in text_lower][:6],
        "guidance_signals": [ph for ph in GUIDANCE_PHRASES if ph in text_lower][:4],
        "chunks_analyzed": 1
    }