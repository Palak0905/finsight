# Lightweight keyword-based sentiment analyzer
# No PyTorch/transformers needed — works on Render free tier

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

def analyze_sentiment(text: str) -> dict:
    """Keyword-based financial sentiment analyzer."""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    # Count keyword hits
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_hits = sum(1 for kw in RISK_KEYWORDS if kw in text_lower)

    # Normalize scores
    pos_score = min(pos_hits / 10, 1.0)
    neg_score = min(neg_hits / 10, 1.0)
    neu_score = max(0, 1.0 - pos_score - neg_score)

    composite = round(pos_score - neg_score, 4)

    if composite > 0.05:
        label = "positive"
        confidence = round(pos_score * 100, 1)
    elif composite < -0.05:
        label = "negative"
        confidence = round(neg_score * 100, 1)
    else:
        label = "neutral"
        confidence = round(neu_score * 100, 1)

    found_risks = [kw for kw in RISK_KEYWORDS if kw in text_lower][:6]
    found_positives = [kw for kw in POSITIVE_KEYWORDS if kw in text_lower][:6]
    found_guidance = [ph for ph in GUIDANCE_PHRASES if ph in text_lower][:4]

    return {
        "score": composite,
        "label": label,
        "confidence": round(confidence, 1),
        "positive_pct": round(pos_score * 100, 1),
        "negative_pct": round(neg_score * 100, 1),
        "neutral_pct": round(neu_score * 100, 1),
        "risk_keywords": found_risks,
        "positive_keywords": found_positives,
        "guidance_signals": found_guidance,
        "chunks_analyzed": 1
    }