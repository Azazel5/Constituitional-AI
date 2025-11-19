#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --mem=80g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-03:00:00
#SBATCH --job-name=Final_Eval_Scoring
#SBATCH --output=logs/scoring_job.out
#SBATCH --error=logs/scoring_job.err

# --- 1. CONFIGURATION ---
# Use absolute path to be safe, or ensure you run sbatch from the project root
PYTHON_SCRIPT="$(pwd)/run_scoring.py" 
OLLAMA_DIR_PATH="/cluster/home/supadh03/ollama"

# Export API Key
export OPENAI_API_KEY=""

# --- 2. DEFINE SCRATCH & CACHE PATHS ---
SCRATCH_ROOT="/tmp/${SLURM_JOB_ID}_scoring"
mkdir -p "${SCRATCH_ROOT}"

# Define variables FIRST
export HF_HOME="${SCRATCH_ROOT}/hf_cache"
export TORCH_HOME="${SCRATCH_ROOT}/torch_cache"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/pip_cache"
export SENTENCE_TRANSFORMERS_HOME="${SCRATCH_ROOT}/sent_trans_cache"
export OLLAMA_HOME="${SCRATCH_ROOT}/ollama_home"
export OLLAMA_MODELS="${OLLAMA_HOME}/models"
export OLLAMA_HOST="127.0.0.1:11434"

# THEN create directories
mkdir -p "${HF_HOME}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${PIP_CACHE_DIR}"
mkdir -p "${SENTENCE_TRANSFORMERS_HOME}"
mkdir -p "${OLLAMA_HOME}"

echo "--- STARTING JOB ---"

# --- 3. SETUP OLLAMA (Copy Directory) ---
if [ ! -d "${OLLAMA_DIR_PATH}" ]; then
    echo "CRITICAL ERROR: Ollama directory NOT found at ${OLLAMA_DIR_PATH}"
    exit 1
fi

# Save original directory so we can return later
ORIGINAL_DIR=$(pwd)

cd "${SCRATCH_ROOT}"
echo "Copying Ollama directory to scratch..."
cp -r "${OLLAMA_DIR_PATH}" .

# Add the 'lib' folder to the library path
export LD_LIBRARY_PATH="${SCRATCH_ROOT}/ollama/lib:$LD_LIBRARY_PATH"
chmod +x "${SCRATCH_ROOT}/ollama/bin/ollama"

# Start Server
echo "Starting Ollama server..."
./ollama/bin/ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

# Wait for Server
echo "Waiting for Ollama to start..."
MAX_RETRIES=60
COUNT=0
while ! curl -s http://127.0.0.1:11434 > /dev/null; do
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "TIMEOUT: Ollama failed to start."
        echo "--- OLLAMA LOGS ---"
        cat ollama_server.log
        kill $OLLAMA_PID
        exit 1
    fi
    sleep 1
    ((COUNT++))
    
done
echo "Ollama is up!"

# Pull Judge Model
echo "Pulling llama3..."
./ollama/bin/ollama pull llama3

# --- 4. RUN PYTHON SCRIPT ---
echo "Starting Python scoring..."

# Return to the directory where the responses.csv files are!
cd "${ORIGINAL_DIR}" 

# Run the script (it will find csvs in current dir)
python -u "${PYTHON_SCRIPT}"

# --- 5. CLEAN UP ---
echo "Cleaning up..."
kill $OLLAMA_PID
rm -rf "${SCRATCH_ROOT}"

echo "Job finished successfully."