FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the benchmark repository
RUN git clone https://github.com/Zongru-Li/Survey-and-Benchmarks-of-DL-for-Molecular-Property-Prediction-in-the-Foundation-Model-Era.git .

# Install PyTorch (CUDA enabled)
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install DGL (matching CUDA 12.4 since cu128 wheel might not exist for DGL, or we use standard)
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# Install PyG and dependencies
RUN pip install --no-cache-dir torch_geometric
RUN pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Install benchmark requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install web server dependencies
RUN pip install --no-cache-dir fastapi uvicorn

# Copy our evaluator service into the correct place
COPY evaluator_service.py src/

# Expose port 8080 for Cloud Run
EXPOSE 8080

ENTRYPOINT ["uvicorn", "src.evaluator_service:app", "--host", "0.0.0.0", "--port", "8080"]
