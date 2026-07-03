import requests
from bs4 import BeautifulSoup
import time

HEADERS = {"User-Agent": "FinSight research tool palak@example.com"}

# Hardcoded CIK map for common tickers (fast + reliable fallback)
KNOWN_CIKS = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "GOOG":  "0001652044",
    "TSLA":  "0001318605",
    "NVDA":  "0001045810",
    "AMZN":  "0001018724",
    "META":  "0001326801",
    "NFLX":  "0001065280",
    "AMD":   "0000002488",
}

def get_cik(ticker: str) -> str:
    """Get SEC CIK number for a ticker symbol."""
    ticker = ticker.upper()

    # Check hardcoded map first (instant)
    if ticker in KNOWN_CIKS:
        return KNOWN_CIKS[ticker]

    # Fallback: search SEC company tickers JSON
    mapping_url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(mapping_url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker:
            cik = str(entry["cik_str"]).zfill(10)
            return cik

    raise ValueError(f"CIK not found for ticker: {ticker}. Try AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, META.")


def fetch_edgar_transcripts(ticker: str, limit: int = 5) -> list:
    """
    Fetch recent 10-K or 10-Q filings from SEC EDGAR for a given ticker.
    Returns list of dicts with keys: text, date, type
    """
    cik = get_cik(ticker)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(submissions_url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    subs = resp.json()

    filings = subs.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])

    results = []
    count = 0
    for form, date, accession in zip(forms, dates, accessions):
        if count >= limit:
            break
        if form not in ("10-K", "10-Q"):
            continue

        accession_clean = accession.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession_clean}/{accession}-index.htm"
        )
        try:
            idx_resp = requests.get(index_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(idx_resp.text, "lxml")
            doc_link = None
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.endswith(".htm") and "index" not in href.lower():
                    doc_link = "https://www.sec.gov" + href
                    break
            if not doc_link:
                continue

            doc_resp = requests.get(doc_link, headers=HEADERS, timeout=15)
            doc_soup = BeautifulSoup(doc_resp.text, "lxml")
            for tag in doc_soup(["script", "style", "table"]):
                tag.decompose()
            text = doc_soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())[:8000]

            if len(text) > 200:
                results.append({"text": text, "date": date, "type": form})
                count += 1
            time.sleep(0.5)
        except Exception:
            continue

    return results