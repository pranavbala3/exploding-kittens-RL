#!/bin/bash
# Quick setup script for Exploding Kittens RL

echo "🎮 Exploding Kittens RL - Setup"
echo "================================"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check for python alias issue
PYTHON_VERSION=$(python --version 2>&1)
if [[ $PYTHON_VERSION == *"2.7"* ]]; then
    echo "⚠️  WARNING: 'python' alias points to Python 2.7"
    echo "   Use 'python3' instead of 'python' for all commands"
    echo ""
fi

echo ""
echo "📦 Installed packages:"
python3 -c "import torch, numpy as np; print(f'  - PyTorch: {torch.__version__}'); print(f'  - NumPy: {numpy.__version__}')"

echo ""
echo "🚀 Ready! Use python3 for all commands:"
echo "  - Train: python3 train.py"
echo "  - Test: python3 test_setup.py"
echo "  - Evaluate: python3 evaluate.py final_model.pt 100"
echo ""
