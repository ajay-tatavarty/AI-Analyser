"""
NEXUS AI — API Test Suite
Run: python test_api.py
Make sure the server is running: uvicorn app.main:app --reload
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("GEMINI_API_KEY", "your_key_here")


def print_result(endpoint, result):
    print(f"\n{'='*60}")
    print(f"  {endpoint}")
    print('='*60)
    if result.get("success"):
        resp = result.get("response", "")
        print(resp[:600] + ("..." if len(resp) > 600 else ""))
    else:
        print("ERROR:", result)


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("\n✅ Health Check:", r.json())


def test_text():
    payload = {
        "message": 'Analyze this JSON: {"sales": [120, 340, 210, 450, 390], "months": ["Jan","Feb","Mar","Apr","May"]}',
        "api_key": API_KEY,
        "mode": "text",
        "history": [],
    }
    r = requests.post(f"{BASE_URL}/chat/text", json=payload)
    print_result("POST /chat/text — JSON analysis", r.json())


def test_code():
    payload = {
        "message": """
def fibonacci(n):
    if n <= 0: return []
    elif n == 1: return [0]
    result = [0, 1]
    for i in range(2, n):
        result.append(result[-1] + result[-2])
    return result
""",
        "api_key": API_KEY,
        "mode": "code",
    }
    r = requests.post(f"{BASE_URL}/chat/text", json=payload)
    print_result("POST /chat/text — Code analysis", r.json())


def test_url():
    payload = {
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "question": "What are the key applications of AI mentioned?",
        "api_key": API_KEY,
    }
    r = requests.post(f"{BASE_URL}/chat/url", json=payload)
    print_result("POST /chat/url", r.json())


def test_pdf(filepath: str = None):
    if not filepath:
        print("\n⚠️  Skipping PDF test — provide a PDF path as argument")
        return
    with open(filepath, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/chat/pdf",
            files={"file": (os.path.basename(filepath), f, "application/pdf")},
            data={"question": "Extract all data as JSON and provide key insights", "api_key": API_KEY},
        )
    print_result("POST /chat/pdf", r.json())


def test_image(filepath: str = None):
    if not filepath:
        print("\n⚠️  Skipping Image test — provide an image path as argument")
        return
    mime = "image/png" if filepath.endswith(".png") else "image/jpeg"
    with open(filepath, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/chat/image",
            files={"file": (os.path.basename(filepath), f, mime)},
            data={"question": "Analyze this chart and provide insights", "api_key": API_KEY},
        )
    print_result("POST /chat/image", r.json())


def test_excel(filepath: str = None):
    if not filepath:
        # Create a sample CSV in memory for demo
        import io
        csv_content = b"Month,Sales,Units,Region\nJan,45000,320,North\nFeb,52000,380,North\nMar,61000,450,South\nApr,48000,340,South\nMay,70000,510,East"
        r = requests.post(
            f"{BASE_URL}/chat/excel",
            files={"file": ("sample_sales.csv", io.BytesIO(csv_content), "text/csv")},
            data={"question": "Analyze sales trends and provide business insights", "api_key": API_KEY},
        )
    else:
        with open(filepath, "rb") as f:
            ext = filepath.split(".")[-1]
            mime = "text/csv" if ext == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            r = requests.post(
                f"{BASE_URL}/chat/excel",
                files={"file": (os.path.basename(filepath), f, mime)},
                data={"question": "Full data analysis with insights", "api_key": API_KEY},
            )
    print_result("POST /chat/excel", r.json())


if __name__ == "__main__":
    import sys
    print("\n🚀 NEXUS AI — API Test Suite\n")
    
    test_health()
    test_text()
    test_code()
    test_url()
    test_excel()

    # Optional — pass file paths as args
    # test_pdf("sample.pdf")
    # test_image("chart.png")

    print("\n\n✅ All tests complete!")
