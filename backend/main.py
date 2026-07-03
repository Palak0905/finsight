from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os

from backend.scraper import fetch_edgar_transcripts
from backend.nlp import analyze_sentiment
from backend.stock import get_stock_data, correlate_signal
from backend.pdf_parser import extract_text_from_pdf

app = FastAPI(title="FinSight API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("frontend/index.html")

@app.post("/api/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files accepted")
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        if not text or len(text.strip()) < 100:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF")
        result = analyze_sentiment(text)
        return {"filename": file.filename, "analysis": result}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/analyze-ticker/{ticker}")
def analyze_ticker(ticker: str):
    try:
        ticker = ticker.upper().strip()
        transcripts = fetch_edgar_transcripts(ticker, limit=2)
        if not transcripts:
            raise HTTPException(status_code=404, detail=f"No filings found for {ticker}")
        results = []
        for t in transcripts:
            try:
                sentiment = analyze_sentiment(t["text"])
                stock_data = get_stock_data(ticker, t["date"])
                correlation = correlate_signal(sentiment["score"], stock_data)
                results.append({
                    "date": t["date"],
                    "filing_type": t["type"],
                    "sentiment": sentiment,
                    "stock": stock_data,
                    "correlation": correlation
                })
            except Exception as e:
                continue
        if not results:
            raise HTTPException(status_code=500, detail="Analysis failed for all filings")
        return {"ticker": ticker, "results": results}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/stock/{ticker}")
def stock_chart(ticker: str, period: str = "1y"):
    try:
        ticker = ticker.upper().strip()
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            raise HTTPException(status_code=404, detail="No stock data found")
        data = [
            {"date": str(d.date()), "close": round(float(c), 2)}
            for d, c in zip(hist.index, hist["Close"])
        ]
        return {"ticker": ticker, "history": data}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "FinSight is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)