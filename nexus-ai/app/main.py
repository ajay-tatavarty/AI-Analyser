"""
AI Analyser AI — Multi-Modal Chatbot Backend
FastAPI + Google Gemini API
"""

import os
import json
import base64
import mimetypes
import httpx
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── APP SETUP ──
app = FastAPI(
    title="AI Analyser AI — Multi-Modal Intelligence API",
    description="Gemini-powered chatbot backend supporting PDF, Images, Excel, URLs, Code & more",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── MODELS ──
class TextRequest(BaseModel):
    message: str
    api_key: str
    mode: Optional[str] = "text"
    history: Optional[list] = []

class URLRequest(BaseModel):
    url: str
    question: str
    api_key: str

# ── GEMINI HELPER ──
def get_gemini_model(api_key: str, model: str = "gemini-2.5-flash"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model)

def build_system_prompt(mode: str) -> str:
    prompts = {
        "pdf": """You are Ai analyser AI, an expert document analyst.
For PDF content provided:
1. Extract all meaningful text, tables, figures, and metadata
2. Return structured data in JSON format where applicable
3. Provide a concise summary with key bullet-point insights
4. Highlight critical data: dates, names, financial figures, decisions
Format your JSON output clearly and include a 'summary' and 'key_insights' array.""",

        "image": """You are AI Analyser AI, an expert visual data analyst.
For images/charts provided:
1. Describe the visual content in detail
2. If chart/graph: identify type, axes labels, units, data ranges
3. Extract trends, peaks, troughs, anomalies, and patterns
4. Generate actionable business/analytical insights in natural language
5. Estimate data values where possible
Be specific with numbers and percentages.""",

        "excel": """You are AI Analyser AI, an expert data analyst.
For tabular data (Excel/CSV/JSON):
1. Summarize the dataset: shape, columns, data types
2. Compute key statistics: mean, median, min, max, nulls
3. Identify trends, outliers, correlations, and anomalies
4. Return a structured JSON with: summary, statistics, insights
5. Provide 3-5 actionable business recommendations
Always format JSON output with proper structure.""",

        "url": """You are AI Analyser AI, an expert research analyst with RAG capabilities.
For web content provided:
1. Extract and synthesize the most relevant information
2. Structure the response with clear sections
3. Cite specific facts, figures, and data points
4. Provide a concise executive summary
5. List key takeaways and recommendations
Answer the user's specific question based on the fetched content.""",

        "code": """You are AI Analyser AI, an expert software engineer and code analyst.
For code/syntax provided:
1. Identify the programming language and purpose
2. Explain what the code does step by step
3. Identify bugs, security issues, or inefficiencies
4. Suggest improvements with examples
5. Rate code quality and provide a structured review in JSON
Include: language, purpose, issues[], improvements[], quality_score (1-10).""",

        "text": """You are AI Analyser AI, a brilliant multi-modal AI assistant.
Analyze the input carefully:
- If JSON: parse, validate, and summarize key fields with statistics
- If code: review, explain, and suggest improvements  
- If plain text: extract entities, themes, sentiments, and insights
- If question: answer thoroughly with examples and structure
Format responses with clear sections. Use JSON where data is structured.""",
    }
    return prompts.get(mode, prompts["text"])


# ══════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><head><title>AI Analyser AI API</title></head>
    <body style="font-family:monospace;background:#0a0a0f;color:#e8e8f0;padding:40px;">
    <h1 style="color:#7c3aed;">⬡ AI Analyser AI — Multi-Modal Backend</h1>
    <p>API is running. Visit <a href="/docs" style="color:#06b6d4;">/docs</a> for interactive documentation.</p>
    <hr style="border-color:#2a2a3d;">
    <h3>Endpoints:</h3>
    <ul>
      <li>POST /chat/text — Text, code, JSON analysis</li>
      <li>POST /chat/pdf — PDF extraction &amp; insights</li>
      <li>POST /chat/image — Image &amp; chart analysis</li>
      <li>POST /chat/excel — Excel/CSV/JSON data analysis</li>
      <li>POST /chat/url — URL/web content RAG</li>
      <li>POST /chat/multimodal — Combined multi-input</li>
      <li>GET  /health — Health check</li>
    </ul>
    </body></html>
    """

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI Analyser AI", "version": "1.0.0"}


# ── 1. TEXT / CODE / JSON ──
@app.post("/chat/text")
async def chat_text(req: TextRequest):
    """Analyze text, code, JSON strings, or general questions."""
    try:
        model = get_gemini_model(req.api_key)
        system = build_system_prompt(req.mode or "text")

        # Build chat with history
        history = []
        for h in req.history:
            history.append({"role": h["role"], "parts": [h["content"]]})

        chat = model.start_chat(history=history)
        full_prompt = f"{system}\n\nUser input:\n{req.message}"
        response = chat.send_message(full_prompt)

        return {
            "success": True,
            "mode": req.mode,
            "response": response.text,
            "input_type": detect_input_type(req.message),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. PDF ANALYSIS ──
@app.post("/chat/pdf")
async def chat_pdf(
    file: UploadFile = File(...),
    question: str = Form("Extract all data and return as structured JSON with summary"),
    api_key: str = Form(...),
):
    """Extract text, tables, and insights from PDF files."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    try:
        contents = await file.read()

        # Extract text using pdfplumber
        extracted = {"pages": [], "tables": [], "total_pages": 0, "total_chars": 0}

        with pdfplumber.open(BytesIO(contents)) as pdf:
            extracted["total_pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                tables = page.extract_tables() or []
                extracted["pages"].append({
                    "page": i + 1,
                    "text": page_text,
                    "char_count": len(page_text),
                })
                for tbl in tables:
                    extracted["tables"].append({
                        "page": i + 1,
                        "data": tbl
                    })
                extracted["total_chars"] += len(page_text)

        full_text = "\n\n".join([p["text"] for p in extracted["pages"]])

        # Send to Gemini
        model = get_gemini_model(api_key)
        system = build_system_prompt("pdf")
        prompt = f"""{system}

PDF Details:
- Total pages: {extracted['total_pages']}
- Total characters: {extracted['total_chars']}
- Tables found: {len(extracted['tables'])}

Extracted Text:
{full_text[:25000]}

Tables (raw):
{json.dumps(extracted['tables'][:5], indent=2)}

User Question: {question}

Respond with structured JSON + natural language insights."""

        response = model.generate_content(prompt)

        return {
            "success": True,
            "mode": "pdf",
            "filename": file.filename,
            "metadata": {
                "pages": extracted["total_pages"],
                "tables_found": len(extracted["tables"]),
                "characters_extracted": extracted["total_chars"],
            },
            "response": response.text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. IMAGE / CHART ANALYSIS ──
@app.post("/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    question: str = Form("Analyze this image and provide detailed insights"),
    api_key: str = Form(...),
):
    """Analyze images and charts using Gemini Vision."""
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")

    try:
        contents = await file.read()
        b64 = base64.b64encode(contents).decode()

        model = get_gemini_model(api_key)
        system = build_system_prompt("image")

        import google.generativeai as genai_module
        image_part = {
            "mime_type": file.content_type,
            "data": b64
        }

        prompt = f"{system}\n\nUser question: {question}"
        response = model.generate_content([prompt, {"inline_data": image_part}])

        return {
            "success": True,
            "mode": "image",
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size_kb": round(len(contents) / 1024, 2),
            "response": response.text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. EXCEL / CSV / JSON DATA ──
@app.post("/chat/excel")
async def chat_excel(
    file: UploadFile = File(...),
    question: str = Form("Provide a full data analysis with key insights"),
    api_key: str = Form(...),
):
    """Analyze Excel, CSV, or JSON data files."""
    filename = file.filename.lower()
    contents = await file.read()

    try:
        # Parse file
        stats = {}
        preview_json = ""

        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(contents))
        elif filename.endswith(".json"):
            data = json.loads(contents.decode("utf-8"))
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data]) if not any(isinstance(v, list) for v in data.values()) \
                     else pd.DataFrame(data)
            else:
                raise ValueError("Unsupported JSON structure")
        else:
            raise HTTPException(status_code=400, detail="Supported: .csv, .xlsx, .xls, .json")

        # Build stats
        stats = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_summary": {}
        }

        num_cols = df.select_dtypes(include="number")
        if not num_cols.empty:
            desc = num_cols.describe().round(3)
            stats["numeric_summary"] = desc.to_dict()

        preview_json = df.head(20).to_json(orient="records", date_format="iso")

        # Send to Gemini
        model = get_gemini_model(api_key)
        system = build_system_prompt("excel")
        prompt = f"""{system}

Dataset Statistics:
{json.dumps(stats, indent=2, default=str)}

First 20 rows (JSON):
{preview_json}

User Question: {question}

Provide:
1. A structured JSON with: summary, statistics, key_metrics, insights[], recommendations[]
2. Natural language analysis below the JSON"""

        response = model.generate_content(prompt)

        return {
            "success": True,
            "mode": "excel",
            "filename": file.filename,
            "dataset_stats": stats,
            "response": response.text,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. URL / WEB CONTENT (RAG) ──
@app.post("/chat/url")
async def chat_url(req: URLRequest):
    """Fetch URL content and answer questions using RAG approach."""
    try:
        # Fetch URL content
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AI AnalyserAI/1.0; research-bot)"
        }
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(req.url, headers=headers)
            resp.raise_for_status()
            raw_html = resp.text
            content_type = resp.headers.get("content-type", "")

        # Parse content
        if "html" in content_type:
            soup = BeautifulSoup(raw_html, "html.parser")
            # Remove scripts, styles, nav
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            page_text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string if soup.title else req.url
        else:
            page_text = raw_html
            title = req.url

        # Truncate for API limits
        page_text = page_text[:20000]

        model = get_gemini_model(req.api_key)
        system = build_system_prompt("url")
        prompt = f"""{system}

Source URL: {req.url}
Page Title: {title}
Content Length: {len(page_text)} characters

--- FETCHED CONTENT START ---
{page_text}
--- FETCHED CONTENT END ---

User Question: {req.question}

Provide a structured, insightful answer based strictly on the fetched content."""

        response = model.generate_content(prompt)

        return {
            "success": True,
            "mode": "url",
            "source": req.url,
            "title": title,
            "content_length": len(page_text),
            "response": response.text,
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. MULTIMODAL (Combined) ──
@app.post("/chat/multimodal")
async def chat_multimodal(
    api_key: str = Form(...),
    message: str = Form("Analyze all attached content and provide insights"),
    mode: str = Form("text"),
    files: list[UploadFile] = File(default=[]),
    urls: str = Form(default=""),  # comma-separated
):
    """
    Multi-input endpoint: combine text + multiple files + URLs in one request.
    Supports PDF, images, Excel, CSV, JSON, code files simultaneously.
    """
    try:
        model = get_gemini_model(api_key)
        system = build_system_prompt(mode)
        parts = [system]
        metadata = []

        # Handle URLs
        if urls.strip():
            url_list = [u.strip() for u in urls.split(",") if u.strip()]
            for url in url_list:
                try:
                    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                        r = await client.get(url, headers={"User-Agent": "AI AnalyserAI/1.0"})
                        soup = BeautifulSoup(r.text, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer"]):
                            tag.decompose()
                        text = soup.get_text(separator="\n", strip=True)[:8000]
                        parts.append(f"[URL: {url}]\n{text}")
                        metadata.append({"type": "url", "source": url})
                except Exception as e:
                    parts.append(f"[URL: {url}] — Failed to fetch: {str(e)}")

        # Handle files
        image_parts = []
        for file in files:
            if not file.filename:
                continue
            content = await file.read()
            fname = file.filename.lower()
            ct = file.content_type or ""

            if ct.startswith("image/"):
                # Vision input
                b64 = base64.b64encode(content).decode()
                image_parts.append({"inline_data": {"mime_type": ct, "data": b64}})
                metadata.append({"type": "image", "filename": file.filename})

            elif fname.endswith(".pdf"):
                with pdfplumber.open(BytesIO(content)) as pdf:
                    pdf_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                parts.append(f"[PDF: {file.filename}]\n{pdf_text[:10000]}")
                metadata.append({"type": "pdf", "filename": file.filename, "pages": len(pdf.pages)})

            elif fname.endswith((".xlsx", ".xls", ".csv")):
                df = pd.read_excel(BytesIO(content)) if fname.endswith((".xlsx", ".xls")) \
                     else pd.read_csv(BytesIO(content))
                parts.append(f"[EXCEL/CSV: {file.filename}]\nShape: {df.shape}\nColumns: {df.columns.tolist()}\nPreview:\n{df.head(10).to_string()}")
                metadata.append({"type": "excel", "filename": file.filename, "shape": list(df.shape)})

            elif fname.endswith(".json"):
                data = json.loads(content.decode())
                parts.append(f"[JSON: {file.filename}]\n{json.dumps(data, indent=2)[:8000]}")
                metadata.append({"type": "json", "filename": file.filename})

            else:
                # Text / code file
                try:
                    text = content.decode("utf-8", errors="replace")
                    parts.append(f"[FILE: {file.filename}]\n```\n{text[:8000]}\n```")
                    metadata.append({"type": "code", "filename": file.filename})
                except Exception:
                    metadata.append({"type": "binary", "filename": file.filename})

        # Compose final prompt
        parts.append(f"\nUser Request: {message}")
        full_prompt = "\n\n".join(str(p) for p in parts)

        # Call Gemini — include images if any
        if image_parts:
            content_parts = [full_prompt] + image_parts
            response = model.generate_content(content_parts)
        else:
            response = model.generate_content(full_prompt)

        return {
            "success": True,
            "mode": "multimodal",
            "inputs_processed": metadata,
            "response": response.text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── UTILITY ──
def detect_input_type(text: str) -> str:
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return "json"
        except Exception:
            pass
    code_keywords = ["def ", "import ", "class ", "function ", "const ", "var ", "let ", "SELECT ", "CREATE "]
    if any(kw in text for kw in code_keywords):
        return "code"
    return "text"
