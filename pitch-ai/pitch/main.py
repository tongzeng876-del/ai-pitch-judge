import os
import shutil
import json
import csv
import io
import asyncio
import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import httpx

# --- 1. CONFIGURATION ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Increase timeout: LlamaParse polling + Whisper + GPT can take 2+ minutes total
client = OpenAI(api_key=api_key, timeout=180.0)
app = FastAPI()

# LlamaParse REST API 配置（直接调用，绕过 SDK 的 Python 3.14 兼容性问题）
LLAMAPARSE_API_BASE = "https://api.cloud.llamaindex.ai/api/v1/parsing"
LLAMAPARSE_ENABLED = bool(llama_cloud_api_key and llama_cloud_api_key != "your_llama_cloud_api_key_here")
if LLAMAPARSE_ENABLED:
    print("✅ LlamaParse enabled (REST API, Markdown mode for charts/tables)")
else:
    print("⚠️ LlamaParse disabled: LLAMA_CLOUD_API_KEY not configured. Using pypdf fallback.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. HELPER FUNCTIONS ---

async def extract_text_with_llamaparse(file_bytes, session_id=""):
    """
    使用 LlamaParse REST API 解析 PDF，返回 Markdown 格式文本。
    直接调用 REST API，绕过 llama-parse SDK 在 Python 3.14 上的兼容性问题。
    
    流程：
    1. 上传 PDF 文件到 LlamaParse API
    2. 轮询等待解析完成
    3. 获取 Markdown 格式的解析结果（图表→表格，复杂布局→结构化文本）
    """
    if not LLAMAPARSE_ENABLED:
        return ""
    
    headers = {
        "Authorization": f"Bearer {llama_cloud_api_key}",
        "Accept": "application/json",
    }
    
    try:
        # Step 1: 上传 PDF 文件
        print(f"[{session_id}] LlamaParse: Uploading PDF...")
        async with httpx.AsyncClient(timeout=120.0) as http:
            upload_resp = await http.post(
                f"{LLAMAPARSE_API_BASE}/upload",
                headers=headers,
                files={"file": ("pitch_deck.pdf", file_bytes, "application/pdf")},
                data={
                    "result_type": "markdown",
                    "language": "en",
                },
            )
            
            if upload_resp.status_code != 200:
                print(f"[{session_id}] LlamaParse Upload Error: {upload_resp.status_code} - {upload_resp.text}")
                return ""
            
            job_id = upload_resp.json().get("id")
            if not job_id:
                print(f"[{session_id}] LlamaParse: No job ID returned")
                return ""
            
            print(f"[{session_id}] LlamaParse: Job created ({job_id}), waiting for result...")
            
            # Step 2: 轮询等待解析完成（最多 90 秒）
            max_wait = 90
            poll_interval = 2
            elapsed = 0
            
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                
                status_resp = await http.get(
                    f"{LLAMAPARSE_API_BASE}/job/{job_id}",
                    headers=headers,
                )
                
                if status_resp.status_code != 200:
                    print(f"[{session_id}] LlamaParse Status Error: {status_resp.status_code}")
                    continue
                
                status = status_resp.json().get("status")
                
                if status == "SUCCESS":
                    break
                elif status in ("ERROR", "FAILED", "CANCELLED"):
                    print(f"[{session_id}] LlamaParse Job Failed: {status}")
                    return ""
                # PENDING / STARTED → 继续等待
            
            if elapsed >= max_wait:
                print(f"[{session_id}] LlamaParse: Timeout after {max_wait}s")
                return ""
            
            # Step 3: 获取 Markdown 结果
            result_resp = await http.get(
                f"{LLAMAPARSE_API_BASE}/job/{job_id}/result/markdown",
                headers=headers,
            )
            
            if result_resp.status_code != 200:
                print(f"[{session_id}] LlamaParse Result Error: {result_resp.status_code}")
                return ""
            
            result_data = result_resp.json()
            
            # 合并所有页面的 Markdown 文本
            pages = result_data.get("markdown", "") or result_data.get("pages", [])
            if isinstance(pages, list):
                full_text = "\n\n".join([p.get("md", "") for p in pages if p.get("md")])
            else:
                full_text = str(pages)
            
            # 限制长度，防止 Token 爆炸
            result = full_text[:15000]
            print(f"[{session_id}] LlamaParse: Success ({len(result)} chars extracted)")
            return result
            
    except Exception as e:
        print(f"[{session_id}] LlamaParse Error: {e}")
        return ""

def extract_text_from_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        # 限制读取页数，防止纯文本过长
        for i, page in enumerate(reader.pages):
            if i >= 20: break 
            text += page.extract_text() + "\n"
        return text[:15000] 
    except Exception as e:
        print(f"PDF Text Error: {e}")
        return ""

def normalize_scores(scores_dict):
    target_map = {
        "problem": ["problem", "evidence"],
        "market": ["market", "user", "target"],
        "solution": ["solution", "product"],
        "biz_model": ["business", "model", "implementation", "biz"],
        "traction": ["traction", "validation"],
        "impact": ["impact", "future"]
    }
    normalized = {k: 4.0 for k in target_map.keys()} 
    if not scores_dict: return normalized

    for key, value in scores_dict.items():
        key_lower = str(key).lower()
        for target, keywords in target_map.items():
            if any(k in key_lower for k in keywords):
                try: normalized[target] = float(value)
                except: pass
                break
    return normalized

def save_to_csv(data: dict):
    filename = 'pitch_judge_results.csv'
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Pitch_Avg', 'Problem', 'Market', 'Solution', 'BizModel', 'Traction', 'Impact'])
            
            p = data['pitch_scores']
            avg_p = sum(p.values()) / len(p) if p else 0
            
            writer.writerow([
                data['timestamp'], f"{avg_p:.1f}",
                p.get('problem',0), p.get('market',0), p.get('solution',0), 
                p.get('biz_model',0), p.get('traction',0), p.get('impact',0)
            ])
    except Exception as e:
        print(f"CSV Error: {e}")

# --- 3. MAIN API ---
@app.post("/analyze_full_session")
async def analyze_full_session(
    pdf_file: UploadFile = File(...),
    pitch_audio: UploadFile = File(...),
    qa_audio: Optional[UploadFile] = File(None)
):
    session_id = str(uuid.uuid4())
    temp_pitch = f"temp_{session_id}_p.wav"
    
    try:
        # 1. 资源处理 (PDF)
        print(f"[{session_id}] Processing PDF...")
        pdf_content = await pdf_file.read()
        
        # 优先使用 LlamaParse 解析（擅长图表/表格），失败则回退到 pypdf
        deck_text = await extract_text_with_llamaparse(pdf_content, session_id)
        if not deck_text.strip():
            print(f"[{session_id}] Falling back to pypdf for text extraction...")
            deck_text = extract_text_from_pdf(pdf_content)
        
        # 2. 音频转录
        print(f"[{session_id}] Processing Audio...")
        with open(temp_pitch, "wb") as f: shutil.copyfileobj(pitch_audio.file, f)

        # 检查音频大小，Whisper 限制 25MB
        file_size_mb = os.path.getsize(temp_pitch) / (1024 * 1024)
        if file_size_mb > 24:
            print(f"[{session_id}] ⚠️ Audio too large ({file_size_mb:.1f}MB). Analysis might fail.")

        try:
            with open(temp_pitch, "rb") as f:
                # 即使音频很长，Whisper 通常也能处理，但如果因为网络超时，这里会捕获
                pitch_text = client.audio.transcriptions.create(model="whisper-1", file=f).text
        except Exception as e:
            print(f"[{session_id}] Audio Error: {e}")
            pitch_text = "(Audio processing failed - analyzing Deck only)"

        # 3. GPT 评分
        print(f"[{session_id}] AI Scoring (Optimized Payload)...")
        
        system_prompt = """
        You are a rigorous Venture Capital Judge.
        Evaluate the "Pitch" (Deck + Speech).
        
        **PART 1: RUBRIC (1-7 Scale)**
        1. Problem & Evidence (1=Unclear -> 7=Rigorous Evidence)
        2. User & Market (1=No User -> 7=Equity Context)
        3. Solution & Product (1=Unclear -> 7=Validated/Ethical AI)
        4. Business Model (1=None -> 7=Credible/Scalable)
        5. Overall Traction (1=None -> 7=Significant Metrics)
        6. Impact & Future (1=Absent -> 7=High Leverage of $20k Prize)

        **PART 2: FEEDBACK FORMAT (CRITICAL)**
        Provide Strengths, Improvements, and Potential Concerns.
        For each point, you MUST follow this structure:
        1. **The Observation**: What did they do?
        2. **The Evidence**: "BECAUSE..." (Quote specific phrases from transcript or data from slides).
        3. **The Detailed Advice**: "THEREFORE..." (Provide specific, actionable steps to improve or leverage this further).

        * **Bad Example**: "Good market analysis because you cited data."
        * **Good Example**: "Strong market analysis BECAUSE you explicitly cited the '8% CAGR in the diabetic market' in Slide 4. **THEREFORE, you should break down this growth by region to show where your immediate entry point lies.**"

        **OUTPUT JSON**:
        {
            "pitch_scores": { "Problem": 1-7, "Market": 1-7, "Solution": 1-7, "Business Model": 1-7, "Traction": 1-7, "Impact": 1-7 },
            "feedback": { "strengths": ["..."], "improvements": ["..."] },
            "potential_concerns": [{"title": "Concern Title", "description": "..."}]
        }
        """

        user_content = f"PITCH TRANSCRIPT:\n{pitch_text}\n\nDECK TEXT:\n{deck_text}"

        completion = client.chat.completions.create(
            model="ft:gpt-4.1-mini-2025-04-14:personal:ai-pitch-judge:DB6W2Iyb",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        raw_data = json.loads(completion.choices[0].message.content)

        final_pitch = normalize_scores(raw_data.get("pitch_scores", {}))

        result = {
            "pitch_scores": final_pitch,
            "feedback": raw_data.get("feedback", {"strengths":[], "improvements":[]}),
            "potential_concerns": raw_data.get("potential_concerns", []),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_to_csv(result)
        return result

    except Exception as e:
        print(f"[{session_id}] ERROR: {e}")
        # 返回 500 错误，让前端知道发生了什么
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_pitch): os.remove(temp_pitch)

# --- 4. SERVE FRONTEND ---
# Serve index.html at root so users access via http://127.0.0.1:8000/
# instead of file:// which causes fetch/CORS issues in browsers
@app.get("/")
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)