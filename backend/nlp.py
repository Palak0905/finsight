from transformers import pipeline
import os

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("Loading sentiment model...")
        _pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None
        )
        print("Model loaded!")
    return _pipeline

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

def chunk_text(text: str, max_words: int = 300) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks[:5]

def analyze_sentiment(text: str) -> dict:
    nlp = get_pipeline()
    chunks = chunk_text(text)

    if not chunks:
        return {"score": 0, "label": "neutral", "confidence": 0,
                "risk_keywords": [], "positive_keywords": [], "guidance_signals": []}

    pos_scores, neg_scores, neu_scores = [], [], []

    for chunk in chunks:
        try:
            result = nlp(chunk[:512])[0]
            for item in result:
                label = item["label"].lower()
                score = item["score"]
                if "pos" in label:
                    pos_scores.append(score)
                elif "neg" in label:
                    neg_scores.append(score)
                else:
                    neu_scores.append(score)
        except Exception:
            continue

    avg_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0
    avg_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0
    avg_neu = sum(neu_scores) / len(neu_scores) if neu_scores else 0

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