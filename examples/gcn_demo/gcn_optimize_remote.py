import asyncio
import logging
import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

# Add alpha_evolve source to path
DEMO_DIR = Path(__file__).parent
REPO_ROOT = DEMO_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import nest_asyncio
from dotenv import load_dotenv

# Load .env from demo dir or repo root
load_dotenv(dotenv_path=DEMO_DIR / ".env")
load_dotenv(dotenv_path=REPO_ROOT / ".env")

from alpha_evolve.client import AlphaEvolveClient
from alpha_evolve.controller import run_controller_loop
from alpha_evolve.experiment import AlphaEvolveExperiment
from alpha_evolve.visualization import get_score

PROJECT_ID = os.getenv("PROJECT_ID")
GE_APP_ID = os.getenv("GE_APP_ID")
LOCATION = os.getenv("LOCATION", "global")
COLLECTION = os.getenv("COLLECTION", "default_collection")
ASSISTANT = os.getenv("ASSISTANT", "default_assistant")
BASE_URL = os.getenv("BASE_URL", "discoveryengine.googleapis.com")

EVALUATOR_URL = os.getenv("EVALUATOR_URL", "https://gcn-evaluator-181550378089.us-central1.run.app/evaluate")
MODEL = "gemini-3.5-flash" 

MAX_PROGRAMS_GENERATED = 20
MAX_PROGRAMS_EVALUATED = 20
EPOCHS = 100
GCN_METRIC = "PearsonR"

logger = logging.getLogger(__name__)

# Read original code for initial program
with open(DEMO_DIR / "gcn_original.py", "r") as f:
    INITIAL_GCN_CODE = f.read()

def gcn_remote_evaluation(program_candidate) -> dict:
    logger.info("Starting GCN remote evaluation (public)")
    files = program_candidate.get("content", {}).get("files", [])
    if not files:
        return {
            "scores": {"scores": []},
            "insights": {"insights": [{"label": "Client Error", "text": "No files in candidate"}]}
        }
        
    payload = {
        "files": files,
        "epochs": EPOCHS
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(EVALUATOR_URL, data=data, headers=headers)
    
    logger.info(f"Sending evaluation request to {EVALUATOR_URL}")
    
    try:
        # Timeout set to 600s (10m)
        with urllib.request.urlopen(req, timeout=600) as response:
            resp_body = response.read().decode("utf-8")
            result_json = json.loads(resp_body)
            logger.info("Received response from evaluator")
            return result_json
            
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8")
        error_msg = f"HTTP Error {e.code}: {err_content}"
        logger.error(error_msg)
        return {
            "scores": {"scores": [{"metric": GCN_METRIC, "score": -1e12}]},
            "insights": {"insights": [{"label": "HTTP Error", "text": error_msg}]}
        }
    except Exception as e:
        error_msg = f"Remote evaluation failed: {e}"
        logger.error(error_msg)
        return {
            "scores": {"scores": [{"metric": GCN_METRIC, "score": -1e12}]},
            "insights": {"insights": [{"label": "Client Exception", "text": error_msg}]}
        }

def main():
    logging.basicConfig(level=logging.INFO)
    
    if not PROJECT_ID or not GE_APP_ID:
        print("ERROR: PROJECT_ID and GE_APP_ID must be set in .env")
        return

    client = AlphaEvolveClient(
        project_id=PROJECT_ID,
        location=LOCATION,
        collection=COLLECTION,
        engine=GE_APP_ID,
        assistant=ASSISTANT,
        base_url=BASE_URL,
    )

    experiment = AlphaEvolveExperiment(
        client,
        gcn_remote_evaluation,
        MAX_PROGRAMS_EVALUATED,
        parallel_evaluation=False, 
    )

    exp_config = {
        "title": "GCN Remote Optimization Demo",
        "problem_description": (
            "Optimize the GCN model architecture defined in src/models/gcn.py. "
            "Modify the layer definitions, message passing, activation functions, "
            "dropout, or pooling to maximize the PearsonR metric on the adme_sol dataset."
        ),
        "program_language": "python",
        "run_settings": {
            "max_programs": MAX_PROGRAMS_GENERATED,
            "concurrency": 2, 
        },
        "generation_settings": {
            "models": [{"name": MODEL, "weight": 1.0}],
        },
    }

    print("Creating experiment...")
    experiment.create_experiment(exp_config)
    
    # Start from original unoptimized code for the demo
    initial_program = {
        "content": {
            "files": [
                {
                    "path": "src/models/gcn.py",
                    "content": INITIAL_GCN_CODE, 
                }
            ]
        },
        "evaluation": {
            "scores": {
                "scores": [{"metric": GCN_METRIC, "score": 0.2465}]
            }
        },
    }

    print("Creating initial program...")
    experiment.create_initial_program(initial_program)
    
    print("Starting experiment...")
    experiment.start_experiment()

    nest_asyncio.apply()
    print("Running evolution loop...")
    asyncio.run(run_controller_loop(experiment, num_samplers=2, num_evaluators=2))

    print("Experiment finished. Retrieving results...")
    list_params = {"order_by": f"{GCN_METRIC} asc"}
    response = experiment.list_programs(params=list_params)

    if response and "alphaEvolvePrograms" in response:
        top_programs = response["alphaEvolvePrograms"]
        top_programs.sort(
            key=lambda p: get_score(p, GCN_METRIC), reverse=True
        )

        print("\nTop Programs:")
        for i, prog in enumerate(top_programs[:5]):
            score_val = get_score(prog, GCN_METRIC)
            print(f"Rank {i + 1} | ID: {prog.get('name')} | Score: {score_val}")
    else:
        print("No programs found.")

if __name__ == "__main__":
    main()
