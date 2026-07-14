import json
import logging
import os
import subprocess
import re
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

PROJECT_ROOT = Path("/app")
GCN_FILE_PATH = PROJECT_ROOT / "src" / "models" / "gcn.py"
GCN_METRIC = "PearsonR"

class FilePayload(BaseModel):
    path: str
    content: str

class EvaluatePayload(BaseModel):
    files: List[FilePayload]
    epochs: int = 100 

# Pre-load original code for restoration
if GCN_FILE_PATH.exists():
    with open(GCN_FILE_PATH, "r") as f:
        ORIGINAL_GCN_CODE = f.read()
else:
    ORIGINAL_GCN_CODE = ""

@app.post("/evaluate")
async def evaluate(payload: EvaluatePayload):
    logger.info("Received evaluation request")
    
    # Find the gcn.py file in the payload
    gcn_file = None
    for f in payload.files:
        if f.path.endswith("gcn.py"):
            gcn_file = f
            break
            
    if not gcn_file:
        raise HTTPException(status_code=400, detail="Missing gcn.py in files payload")
        
    score_value = -1e12
    insights_list = []
    
    # 1. Write candidate code to gcn.py
    try:
        GCN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GCN_FILE_PATH, "w") as f:
            f.write(gcn_file.content)
            
        # 2. Run training script
        cmd = [
            "python3",
            "src/run.py",
            "--config", "configs/gcn.yaml",
            "--dataset", "adme_sol",
            "--epochs", str(payload.epochs),
            "--device", "cuda"
        ]
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            error_msg = f"Training failed with exit code {result.returncode}.\nStderr: {result.stderr}"
            logger.error(error_msg)
            insights_list.append(
                {"label": "Training Error", "text": error_msg}
            )
        else:
            # 3. Parse score from stdout
            stdout = result.stdout
            logger.debug(f"Training stdout:\n{stdout}")
            
            match = re.search(r"mean:\s+([0-9.-]+)", stdout)
            if match:
                score_value = float(match.group(1))
                logger.info(f"Parsed score: {score_value}")
                if score_value != score_value: # NaN check
                     score_value = -1e12
                     insights_list.append(
                         {"label": "NaN Score", "text": "PearsonR was NaN"}
                     )
            else:
                error_msg = "Could not parse mean score from training output."
                logger.error(error_msg)
                insights_list.append(
                    {"label": "Parsing Error", "text": error_msg}
                )
                
    except subprocess.TimeoutExpired as e:
        error_msg = f"Training timed out: {e}"
        logger.error(error_msg)
        insights_list.append(
            {"label": "Timeout", "text": error_msg}
        )
    except Exception as e:
        error_msg = f"Unexpected error during evaluation: {e}"
        logger.error(error_msg)
        insights_list.append(
            {"label": "Unexpected Error", "text": error_msg}
        )
    finally:
        # Restore original code
        if ORIGINAL_GCN_CODE:
            logger.info("Restoring original GCN code")
            with open(GCN_FILE_PATH, "w") as f:
                f.write(ORIGINAL_GCN_CODE)
        
    response_data = {
        "scores": {
            "scores": [
                {"metric": GCN_METRIC, "score": score_value}
            ]
        }
    }
    if insights_list:
        response_data["insights"] = {
            "insights": insights_list
        }
        
    return response_data

@app.get("/health")
def health():
    return {"status": "ok"}
