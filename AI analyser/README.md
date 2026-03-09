Here is a **simple normal README text** you can put in your project.

---

# AI Analyser – Setup Guide

## 1. Clone or Download the Project

Download the project and open the folder in your terminal.

```
AI analyser/
```

---

# 2. Create Virtual Environment

Inside the project folder run:

```
python -m venv venv
```

Activate the environment.

### Windows

```
venv\Scripts\activate
```

### Linux / Mac

```
source venv/bin/activate
```

---

# 3. Install Requirements

Install all required Python libraries.

```
pip install -r requirements.txt
```

---

# 4. Navigate to Backend Folder

```
cd ai-analyser
```

---

# 5. Run FastAPI Backend

Start the server using Uvicorn.

```
uvicorn app.main:app --reload --port 8000
```

Backend will start at:

```
http://127.0.0.1:8000
```

API docs:

```
http://127.0.0.1:8000/docs
```

---

# 6. Open Frontend

The frontend is separate.

Open this file in your browser:

```
ai-chat.html
```

This file will connect to the backend running on **localhost:8000**.

---

# Project Structure

```
AI analyser
│
├── app
│   └── main.py
│
├── uploads
│
├── ai-chat.html
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Notes

* Make sure the backend is running before opening the frontend.
* API key configuration may be required in the backend depending on the AI model used.

---

