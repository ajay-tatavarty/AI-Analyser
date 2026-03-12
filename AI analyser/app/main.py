

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
from pydantic import BaseModel

# ── APP SETUP ──
app = FastAPI(
    title="AI Analyser AI — Multi-Modal Intelligence API",
    description="Gemini-powered chatbot backend supporting PDF, Images, Excel, URLs, Code & more",
    version="3.0.0",
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


# ── SYSTEM PROMPTS ──
def build_system_prompt(mode: str) -> str:
    prompts = {

        # ── TEXT (default) ──
        "text": """You are AI Analyser AI, a helpful and friendly AI assistant.

IMPORTANT — Read the user message carefully and respond appropriately:

1. If the message is a greeting (hi, hello, hey, how are you, etc.):
   → Just reply naturally and conversationally. Say hello back. Ask how you can help.
   → Do NOT analyse it. Do NOT use headers or bullet points.

2. If the message is a simple question:
   → Answer it directly and clearly in plain conversational English.
   → Only use formatting if the answer genuinely needs it.

3. If the message is JSON data:
   → Summarise the key fields and provide insights in plain English.
   → Do NOT just echo the JSON back.

4. If the message is code:
   → Explain what it does, identify issues, suggest improvements.
   → Use ## headers and code blocks where helpful.

5. If the message is a long document, article, or complex text:
   → Use ## headers and bullet points to structure your response.
   → Extract key themes, insights, and important information.

RULE: Match your response style to the input.
Simple input = simple, friendly response.
Complex input = structured, detailed response.
NEVER over-analyse a simple message.""",

        # ── PDF ──
        "pdf": """You are AI Analyser AI, an expert document analyst.

When analysing any PDF document:
1. First identify what type of document this is (resume, report, invoice, contract, research paper, etc.)
2. Present the information in CLEAN, READABLE PLAIN TEXT — NOT raw JSON.
3. Use clear section headers like: ## Personal Info, ## Summary, ## Experience, etc.
4. Use bullet points (•) for lists and sub-items.
5. Bold important labels using **Label:** format.
6. Highlight key data: names, dates, figures, decisions.
7. End with a short ## Key Insights section summarising the most important takeaways.

STRICT RULES:
- Do NOT wrap the entire response in JSON.
- Do NOT output raw curly-brace data structures as your main response.
- Write as a professional analyst presenting findings to a manager.
- Keep it clean, structured, and easy to read.""",

        # ── IMAGE ──
        "image": """You are AI Analyser AI, an expert visual data analyst.

When analysing any image or chart:
1. Start with a one-line description of what the image shows.
2. If it is a chart or graph: identify the type, axes, units, and data ranges.
3. Extract all visible trends, peaks, troughs, anomalies, and patterns.
4. Use clear sections: ## Overview, ## Key Data Points, ## Trends & Patterns, ## Insights.
5. Use bullet points for lists of findings.
6. Estimate specific data values where possible.
7. End with ## Recommendations based on the visual data.

Write in plain, professional English. No raw JSON.""",

        # ── EXCEL / DATA ──
        "excel": """You are AI Analyser AI, an expert data analyst.

When analysing any tabular dataset (Excel, CSV, JSON):
1. ## Dataset Overview: describe what the data is about, rows/columns, date ranges.
2. ## Key Statistics: mean, median, min, max for numeric columns as readable sentences.
3. ## Trends & Patterns: what is going up, down, or unusual.
4. ## Outliers & Anomalies: flag anything significant.
5. ## Business Insights: 3-5 plain-English observations.
6. ## Recommendations: 3 actionable next steps.

Use headers (##), bullet points (•), and bold labels (**Label:**).
Do NOT return raw JSON as your main answer.""",

        # ── URL ──
        "url": """You are AI Analyser AI, an expert research analyst.

When analysing web content:
1. ## Source Summary: a 2-3 sentence overview of what this page is about.
2. ## Key Information: the most relevant facts and data points as bullet points.
3. ## Answer: directly answer the user's specific question.
4. ## Additional Context: any extra relevant info from the page.
5. ## Takeaways: 3-5 key points to remember.

Write in plain, professional English. Use headers and bullet points.
Do NOT return raw JSON.""",

        # ── CODE ──
        "code": """You are AI Analyser AI, an expert software engineer and code reviewer.

When analysing any code:
1. ## Language & Purpose: identify the language and what the code does in 1-2 sentences.
2. ## Step-by-Step Explanation: walk through the code block by block in plain English.
3. ## Issues Found: list bugs, security vulnerabilities, or inefficiencies.
4. ## Improvements: suggest better ways with examples.
5. ## Code Quality: rate the code 1-10 and explain why.

Use headers (##) and bullet points (•).
Wrap code examples in triple backticks.
Write clearly enough for a junior developer to understand.""",

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
    return {"status": "ok", "service": "AI Analyser AI", "version": "3.0.0"}


# ── 1. TEXT / CODE / JSON ──
@app.post("/chat/text")
async def chat_text(req: TextRequest):
    """Analyze text, code, JSON strings, or general questions."""
    try:
        model = get_gemini_model(req.api_key)

        # Build chat history
        history = []
        for h in req.history:
            history.append({"role": h["role"], "parts": [h["content"]]})

        chat = model.start_chat(history=history)

        # If simple greeting — use a minimal prompt so Gemini doesn't over-analyse
        if is_simple_greeting(req.message):
            full_prompt = (
                "You are AI Analyser AI, a friendly AI assistant. "
                f"The user said: '{req.message}'. "
                "Reply naturally and conversationally in 1-2 sentences. "
                "Do not analyse it. Do not use headers or bullet points. "
                "Just say hello and ask how you can help."
            )
        else:
            system = build_system_prompt(req.mode or "text")
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
    question: str = Form(
        "Extract and present all information in a clean, readable format "
        "with clear sections and bullet points. Do not return raw JSON."
    ),
    api_key: str = Form(...),
):
    """Extract text, tables, and insights from PDF files."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    try:
        contents = await file.read()
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
                    extracted["tables"].append({"page": i + 1, "data": tbl})
                extracted["total_chars"] += len(page_text)

        full_text = "\n\n".join([p["text"] for p in extracted["pages"]])
        doc_type_hint = detect_doc_type(full_text)

        model = get_gemini_model(api_key)
        system = build_system_prompt("pdf")

        prompt = f"""{system}

Document Details:
- Detected type: {doc_type_hint}
- Total pages: {extracted['total_pages']}
- Total characters: {extracted['total_chars']}
- Tables found: {len(extracted['tables'])}

Extracted Text:
{full_text[:25000]}

{f"Tables found:{chr(10)}{format_tables(extracted['tables'][:3])}" if extracted['tables'] else ""}

User Question / Instruction: {question}

Present your response in clean, well-structured plain text with ## headers and bullet points.
Do NOT return a raw JSON object as your answer."""

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
    question: str = Form(
        "Analyze this image in detail. Describe what you see, extract all "
        "data and trends, and provide clear insights in plain readable text."
    ),
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

        image_part = {"mime_type": file.content_type, "data": b64}
        prompt = (
            f"{system}\n\nUser question: {question}\n\n"
            "Respond in clean, structured plain text with ## headers and bullet points. No raw JSON."
        )

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
    question: str = Form(
        "Provide a full data analysis with key statistics, trends, "
        "and business insights in plain readable format."
    ),
    api_key: str = Form(...),
):
    """Analyze Excel, CSV, or JSON data files."""
    filename = file.filename.lower()
    contents = await file.read()

    try:
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

        stats = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        }

        num_cols = df.select_dtypes(include="number")
        numeric_summary = {}
        if not num_cols.empty:
            numeric_summary = num_cols.describe().round(3).to_dict()

        preview = df.head(20).to_string(index=False)

        model = get_gemini_model(api_key)
        system = build_system_prompt("excel")

        prompt = f"""{system}

Dataset Info:
- Rows: {stats['rows']} | Columns: {stats['columns']}
- Column names: {', '.join(stats['column_names'])}
- Null values: {stats['null_counts']}
- Numeric stats: {json.dumps(numeric_summary, indent=2, default=str)}

First 20 rows preview:
{preview}

User Question: {question}

Respond in clean, structured plain text with ## headers and bullet points.
Do NOT return raw JSON as your main answer."""

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
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AIAnalyserAI/3.0; research-bot)"}

        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(req.url, headers=headers)
            resp.raise_for_status()
            raw_html = resp.text
            content_type = resp.headers.get("content-type", "")

        if "html" in content_type:
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            page_text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string if soup.title else req.url
        else:
            page_text = raw_html
            title = req.url

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

Respond in clean, structured plain text with ## headers and bullet points.
Answer the user's question clearly and directly."""

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
    message: str = Form("Analyze all attached content and provide clear, readable insights"),
    mode: str = Form("text"),
    files: list[UploadFile] = File(default=[]),
    urls: str = Form(default=""),
):
    """Multi-input: combine text + multiple files + URLs in one request."""
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
                        r = await client.get(url, headers={"User-Agent": "AIAnalyserAI/3.0"})
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
                parts.append(
                    f"[DATA FILE: {file.filename}]\n"
                    f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    f"Columns: {', '.join(df.columns.tolist())}\n"
                    f"Preview:\n{df.head(10).to_string(index=False)}"
                )
                metadata.append({"type": "excel", "filename": file.filename, "shape": list(df.shape)})

            elif fname.endswith(".json"):
                data = json.loads(content.decode())
                parts.append(f"[JSON: {file.filename}]\n{json.dumps(data, indent=2)[:8000]}")
                metadata.append({"type": "json", "filename": file.filename})

            else:
                try:
                    text = content.decode("utf-8", errors="replace")
                    parts.append(f"[FILE: {file.filename}]\n```\n{text[:8000]}\n```")
                    metadata.append({"type": "code", "filename": file.filename})
                except Exception:
                    metadata.append({"type": "binary", "filename": file.filename})

        parts.append(
            f"\nUser Request: {message}\n\n"
            "Respond in clean, structured plain text with ## section headers and bullet points. "
            "Do NOT return raw JSON as your main response."
        )
        full_prompt = "\n\n".join(str(p) for p in parts)

        if image_parts:
            response = model.generate_content([full_prompt] + image_parts)
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


# ══════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════

def is_simple_greeting(text: str) -> bool:
    """Returns True if the message is just a casual greeting."""
    greetings = [
        "hi", "hello", "hey", "hii", "helo", "howdy", "sup", "wassup",
        "good morning", "good afternoon", "good evening", "good night",
        "how are you", "how r u", "how are u", "whats up", "what's up",
        "hi there", "hello there", "hey there", "greetings", "yo",
        "hiya", "heya", "morning", "evening", "afternoon",
        "hi!", "hello!", "hey!", "hii!", "yo!"
    ]
    cleaned = text.strip().lower().rstrip("!.,? ")
    return cleaned in greetings


def detect_input_type(text: str) -> str:
    """Detect whether input is JSON, code, or plain text."""
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return "json"
        except Exception:
            pass
    code_keywords = [
        "def ", "import ", "class ", "function ", "const ",
        "var ", "let ", "SELECT ", "CREATE ", "#include", "public static"
    ]
    if any(kw in text for kw in code_keywords):
        return "code"
    return "text"


def detect_doc_type(text: str) -> str:
    """Guess the document type from its text content."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["experience", "education", "skills", "resume", "curriculum vitae", "linkedin"]):
        return "Resume / CV"
    if any(w in text_lower for w in ["invoice", "bill to", "total amount", "payment due", "subtotal"]):
        return "Invoice / Financial Document"
    if any(w in text_lower for w in ["abstract", "introduction", "methodology", "references", "conclusion"]):
        return "Research Paper / Academic Document"
    if any(w in text_lower for w in ["agreement", "contract", "terms", "party", "clause", "whereas"]):
        return "Legal Contract / Agreement"
    if any(w in text_lower for w in ["revenue", "profit", "loss", "balance sheet", "quarterly", "fiscal"]):
        return "Financial Report"
    if any(w in text_lower for w in ["chapter", "section", "table of contents", "appendix"]):
        return "Report / Manual"
    return "General Document"


def format_tables(tables: list) -> str:
    """Format extracted PDF tables into readable text."""
    if not tables:
        return ""
    result = []
    for t in tables:
        result.append(f"Page {t['page']} Table:")
        if t['data']:
            for row in t['data']:
                result.append("  " + " | ".join(str(cell or "") for cell in row))
    return "\n".join(result)
