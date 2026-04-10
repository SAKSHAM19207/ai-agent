# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Jaoe Environment.

This module creates an HTTP server that exposes the JaoeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import JaoeAction, JaoeObservation
    from .jaoe_environment import JaoeEnvironment
except ImportError:
    from models import JaoeAction, JaoeObservation
    from server.jaoe_environment import JaoeEnvironment


# Create the app with web interface and README integration
app = create_app(
    JaoeEnvironment,
    JaoeAction,
    JaoeObservation,
    env_name="jaoe",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

import os
from fastapi import Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import httpx
from huggingface_hub import AsyncInferenceClient
import random
import json
from dotenv import load_dotenv
import io
from pypdf import PdfReader

load_dotenv()

HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

def get_hf_client():
    hf_token = os.getenv("HF_TOKEN")
    return AsyncInferenceClient(model=HF_MODEL, token=hf_token) if hf_token else None

# --- NEW REST API ROUTES FOR INTERACTIVE DASHBOARD ---

MOCK_JOBS = [
    {"id": 1, "title": "Senior Python Developer", "company": "Meta", "match": 85, "url": "https://meta.com/careers"},
    {"id": 2, "title": "Frontend Ninja (React)", "company": "StartupX", "match": 40, "url": "https://startupx.com/jobs"},
    {"id": 3, "title": "Machine Learning Engineer", "company": "OpenAI", "match": 92, "url": "https://openai.com/careers"},
]

@app.get("/api/jobs")
async def get_jobs():
    try:
        async with httpx.AsyncClient(timeout=10.0) as hc:
            resp = await hc.get("https://remotive.com/api/remote-jobs?category=software-dev&limit=5")
            if resp.status_code == 200:
                data = resp.json()
                real_jobs = []
                for job in data.get("jobs", [])[:5]:
                    real_jobs.append({
                        "id": job["id"],
                        "title": job["title"],
                        "company": job["company_name"],
                        "match": random.randint(40, 98),
                        "url": job.get("url", "https://google.com")
                    })
                if len(real_jobs) > 0:
                    return {"jobs": real_jobs}
    except Exception:
        pass
    return {"jobs": MOCK_JOBS}

@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return {"text": text.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/generate_cover_letter")
async def generate_cover_letter(req: Request):
    client = get_hf_client()
    data = await req.json()
    job_title = data.get("title", "this role")
    company = data.get("company", "your company")
    
    if not client:
        return {"cover_letter": f"Dear Hiring Manager,\n\n(No HF_TOKEN detected in terminal!)\nI am ecstatic to apply for {job_title} at {company}!"}
        
    try:
        prompt = f"Write a professional 100-word cover letter exclusively for the role of {job_title} at {company}."
        resp = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        return {"cover_letter": resp.choices[0].message.content}
    except Exception as e:
        return {"cover_letter": f"HF Error: {str(e)}"}

@app.post("/api/optimize_resume")
async def optimize_resume(req: Request):
    client = get_hf_client()
    data = await req.json()
    raw_text = data.get("resume_text", "")
    cats_score = random.randint(65, 95)
    
    if not client or not raw_text:
        return {
            "optimized": "• Optimized for ATS via heuristic rules.\n• Included action verbs and quantified metrics.\n• Formatted cleanly without complex tables.",
            "ats_score": cats_score
        }
        
    try:
        prompt = f"Rewrite this resume into 3 highly impactful bullet points optimizing for ATS. Focus on metrics: {raw_text}"
        resp = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        return {
            "optimized": resp.choices[0].message.content,
            "ats_score": cats_score
        }
    except Exception as e:
        return {
            "optimized": f"HF Error: {str(e)}",
            "ats_score": cats_score
        }

@app.post("/api/prep_interview")
async def prep_interview(req: Request):
    client = get_hf_client()
    data = await req.json()
    job_title = data.get("title", "this role")
    
    fallback_qa = [{"Q": "How do you handle scaling?", "A": "Provide an HF_TOKEN to see actual AI prep."}]
    
    if not client:
        return {"qa": fallback_qa}
        
    try:
        prompt = f"Generate 5 tricky interview questions and answers for a {job_title} position. You must reply strictly with a raw JSON block containing a 'qa' array where each object has 'Q' and 'A'."
        resp = await client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        
        # Extremely basic JSON extraction for Hugging Face output differences
        res_text = resp.choices[0].message.content
        
        def safely_get_qa(parsed_data):
            if isinstance(parsed_data, list):
                return parsed_data
            elif isinstance(parsed_data, dict):
                return parsed_data.get("qa", fallback_qa)
            return fallback_qa

        try:
            return {"qa": safely_get_qa(json.loads(res_text))}
        except Exception:
            # Fallback if the model outputs markdown wrapper or malformed JSON
            text_to_parse = res_text
            if "```json" in text_to_parse:
                text_to_parse = text_to_parse.split("```json")[1].split("```")[0].strip()
            elif "```" in text_to_parse:
                text_to_parse = text_to_parse.split("```")[1].strip()
            
            try:
                return {"qa": safely_get_qa(json.loads(text_to_parse))}
            except Exception:
                # Ultimate graceful fallback: just display the raw text directly in UI
                return {"qa": [{"Q": "AI Generated Prep", "A": res_text}]}
            
    except Exception as e:
        return {"qa": [{"Q": "HF Error", "A": str(e)}]}

from fastapi.responses import RedirectResponse

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/dashboard/")

frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
print(f"[*] Mounting dashboard from: {frontend_dist}")
if os.path.isdir(frontend_dist):
    print("[*] Dashboard directory found, mounting at /dashboard")
    app.mount("/dashboard", StaticFiles(directory=frontend_dist, html=True), name="dashboard")
else:
    print("[!] Dashboard directory NOT FOUND. Static files will not be served.")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m jaoe.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn jaoe.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
