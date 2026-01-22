import os
import shutil
import json
import csv
import io
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

# --- 1. 配置 ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("⚠️ WARNING: OPENAI_API_KEY not found.")

client = OpenAI(api_key=api_key)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 辅助函数 ---
def extract_text_from_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text[:12000] 
    except Exception as e:
        print(f"PDF Error: {e}")
        return "Error reading PDF file."

def save_to_csv(data: dict):
    filename = 'ward_infinity_scores.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 更新表头以匹配新的6个维度
            writer.writerow(['Timestamp', 'Pitch_Problem', 'Pitch_Market', 'Pitch_Solution', 'Pitch_BizModel', 'Pitch_Traction', 'Pitch_Impact', 'Deck_Preview'])
        
        # 写入 Pitch 的得分作为主要记录
        writer.writerow([
            data['timestamp'],
            data['pitch_scores'].get('problem', 0),
            data['pitch_scores'].get('market', 0),
            data['pitch_scores'].get('solution', 0),
            data['pitch_scores'].get('biz_model', 0),
            data['pitch_scores'].get('traction', 0),
            data['pitch_scores'].get('impact', 0),
            data['deck_text'][:50].replace('\n', ' ') + "..."
        ])

# --- 3. 核心分析接口 ---
@app.post("/analyze_full_session")
async def analyze_full_session(
    pdf_file: UploadFile = File(...),
    pitch_audio: UploadFile = File(...),
    qa_audio: UploadFile = File(...)
):
    temp_pitch = f"temp_pitch_{datetime.now().strftime('%H%M%S')}.wav"
    temp_qa = f"temp_qa_{datetime.now().strftime('%H%M%S')}.wav"
    
    try:
        # A. 提取 PDF
        print("📄 Reading Deck...")
        pdf_content = await pdf_file.read()
        deck_text = extract_text_from_pdf(pdf_content)
        
        # B. 转录 Pitch
        print("🎙️ Transcribing Pitch...")
        with open(temp_pitch, "wb") as buffer:
            shutil.copyfileobj(pitch_audio.file, buffer)
        with open(temp_pitch, "rb") as f:
            pitch_res = client.audio.transcriptions.create(model="whisper-1", file=f)
        pitch_text = pitch_res.text

        # C. 转录 Q&A
        print("🎙️ Transcribing Q&A...")
        with open(temp_qa, "wb") as buffer:
            shutil.copyfileobj(qa_audio.file, buffer)
        with open(temp_qa, "rb") as f:
            qa_res = client.audio.transcriptions.create(model="whisper-1", file=f)
        qa_text = qa_res.text

        # D. GPT-4o 深度评分 (Ward Infinity Rubric)
        print("🧠 Analyzing against Ward Infinity Rubric...")
        
        system_prompt = """
        You are a judge for the "Ward Infinity Showcase 2026". 
        Evaluate the startup based STRICTLY on the official Ward Infinity Rubric.
        
        SCORING SCALE: 1 to 7 (1=Poor, 7=Excellent).
        
        EVALUATION CRITERIA:
        1. Problem & Evidence: Clear problem statement? Credible data (clinical/industry)? Primary/secondary research? Medical validity?
        2. Target User & Market: Defined target user? Market size significance? Understanding of equity/access/community context?
        3. Solution & Product: Solution clear? MVP/Pilot demonstrated? AI used appropriately? Differentiated?
        4. Implementation & Biz Model: Sustainable model? Realistic GTM? Implementation plan? Awareness of regulation/workflow constraints?
        5. Traction & Validation: Meaningful metrics? Partnerships? Funding/Grants? Validation of need?
        6. Impact & Future: Mission/Vision clear? Health/Community impact? Roadmap? Clear use of the $20,000 prize?

        INPUTS:
        - Pitch Deck (PDF content)
        - Spoken Pitch Transcript
        - Q&A Transcript

        TASK:
        Score the "Pitch Performance" (Speech + Deck) and the "Q&A Performance" separately for consistency.
        
        RETURN JSON ONLY:
        {
            "pitch_scores": {
                "problem": (1-7),
                "market": (1-7),
                "solution": (1-7),
                "biz_model": (1-7),
                "traction": (1-7),
                "impact": (1-7)
            },
            "qa_scores": {
                "problem": (1-7),
                "market": (1-7),
                "solution": (1-7),
                "biz_model": (1-7),
                "traction": (1-7),
                "impact": (1-7)
            },
            "audit_flags": [
                {"title": "Missing $20k Plan", "description": "Founder failed to explain how the prize money would be used (Rubric Requirement)."},
                {"title": "Medical Validity Risk", "description": "Claims about health outcomes lack clinical evidence."}
            ],
            "feedback": {
                "strengths": ["Strong community impact", "Clear MVP"],
                "improvements": ["Define use of funds", "Clarify regulatory path"]
            }
        }
        """
        
        user_prompt = f"""
        [PITCH DECK CONTENT]:
        {deck_text}

        [SPOKEN PITCH TRANSCRIPT]:
        {pitch_text}

        [Q&A TRANSCRIPT]:
        {qa_text}
        """

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        analysis = json.loads(completion.choices[0].message.content)
        
        result = {
            "deck_text": deck_text,
            "pitch_text": pitch_text,
            "qa_text": qa_text,
            "pitch_scores": analysis["pitch_scores"],
            "qa_scores": analysis["qa_scores"],
            "audit_flags": analysis["audit_flags"],
            "feedback": analysis["feedback"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_to_csv(result)
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_pitch): os.remove(temp_pitch)
        if os.path.exists(temp_qa): os.remove(temp_qa)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)