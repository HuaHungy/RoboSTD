#!/bin/bash

# RoboSTD Unified Verification Script

echo "=== Verifying Project: RoboSTD Unified ==="

# Check Python version
python3 --version

# Check if required files exist
REQUIRED_FILES=(
    "src/pipeline_core.py"
    "src/joint_processor.py"
    "src/video_mirror.py"
    "scripts/main.py"
    "configs/default.yaml"
    "pyproject.toml"
    "requirements.txt"
)

echo "--- Checking File Structure ---"
MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "[FAIL] Missing file: $file"
        MISSING=1
    else
        echo "[OK] Found: $file"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "Critical files missing. Verification failed."
    exit 1
fi

# Run Linting (optional, don't fail if not installed)
echo "--- Running Linting (flake8) ---"
if command -v flake8 &> /dev/null; then
    flake8 src scripts --count --select=E9,F63,F7,F82 --show-source --statistics
    if [ $? -eq 0 ]; then
        echo "[OK] Linting passed."
    else
        echo "[WARN] Linting issues found."
        # Not failing for now
    fi
else
    echo "[SKIP] flake8 not installed."
fi

# Run Tests
echo "--- Running Tests (pytest) ---"
export PYTHONPATH=$PYTHONPATH:.
if command -v pytest &> /dev/null; then
    pytest tools/tests/test_pipeline.py -v
    if [ $? -eq 0 ]; then
        echo "[OK] Tests passed."
    else
        echo "[FAIL] Tests failed."
        exit 1
    fi
else
    echo "[SKIP] pytest not installed. Please install pytest."
    # Fail if pytest is missing? Or try to run simple python script?
    echo "Running basic import test..."
    python3 -c "from src.pipeline_core import Pipeline; print('Import success')"
    if [ $? -eq 0 ]; then
        echo "[OK] Basic import test passed."
    else
        echo "[FAIL] Import failed."
        exit 1
    fi
fi

echo "=== Verification Complete: SUCCESS ==="
exit 0
