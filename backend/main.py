from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from ml_engine import analyze_data
from nlp_engine import generate_insight

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), analysisType: str = Form(...)):
    df = pd.read_csv(file.file)
    result, model_info = analyze_data(df, analysisType)
    comment = generate_insight(result)
    return {
        "model": model_info,
        "summary": comment,
        "raw_output": result.to_json()
    }
