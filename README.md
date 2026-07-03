# 📈 FinSight — AI-Powered Earnings Call Analyzer

FinSight uses **FinBERT** (a finance-trained NLP model) to analyze earnings call transcripts from SEC EDGAR, extract sentiment signals, and correlate them with stock price movements.

Live Demo:
https://finsight-uy6z.onrender.com


 Features

- 🔍Ticker Analysis — Fetch real SEC 10-K/10-Q filings automatically
- 📄 PDF Upload — Upload any earnings call PDF for instant analysis  
- 🤖 FinBERT NLP — Finance-trained AI for accurate sentiment scoring
- 📊 Stock Correlation— See if sentiment predicted price movement
- 📈Interactive Charts — Sentiment trends + stock price visualization
- ⚡ REST API — Fully documented FastAPI backend


 Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| NLP | FinBERT (HuggingFace Transformers) |
| Data | SEC EDGAR API, yFinance |
| PDF | PDFMiner |
| Frontend | Vanilla JS, HTML5, CSS3, Chart.js |
| Deploy | Render.com |



## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/analyze-ticker/{ticker}` | Analyze SEC filings for a stock |
| POST | `/api/analyze-pdf` | Upload & analyze a PDF |
| GET | `/api/stock/{ticker}` | Get 1-year stock chart data |

---

## 📁 Project Structure

```
finsight/
├── backend/
│   ├── main.py          # FastAPI app + routes
│   ├── nlp.py           # FinBERT sentiment analysis
│   ├── scraper.py       # SEC EDGAR data fetcher
│   ├── stock.py         # yFinance stock data + correlation
│   └── pdf_parser.py    # PDF text extraction
├── frontend/
│   └── index.html       # Full UI (single file)
├── data/                # Local data cache
├── notebooks/           # Jupyter research notebooks
├── requirements.txt
├── run.py               # App entry point
├── render.yaml          # Render deployment config
└── README.md


 🔬 Research Output

This project produces a **research-grade signal** by:
1. Extracting management sentiment from official SEC filings
2. Scoring each filing on a -1 to +1 scale using FinBERT
3. Measuring alignment between sentiment direction and subsequent stock price movement

 📜 License
MIT License — open source, free to use and build upon.
