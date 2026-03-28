#!/bin/bash
# Setup script for MLE interview prep exercises
# Creates a conda environment with all required dependencies

set -e

ENV_NAME="mle_prep"

echo "Setting up conda environment: $ENV_NAME"

# Create the environment if it doesn't exist
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists — skipping creation"
else
    conda create -y -n "$ENV_NAME" python=3.11
    echo "Created environment '$ENV_NAME'"
fi

# Install dependencies
echo "Installing dependencies..."
conda run -n "$ENV_NAME" pip install torch numpy

echo ""
echo "Setup complete."
echo ""
echo "Activate your environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Start exercise 1 (it will fail — that's the point, fix it):"
echo "  cd exercises/user_analytics_pipeline"
echo "  python data_loader.py"
echo ""
echo "Check a solution:"
echo "  python evaluation/self_check.py --exercise 1"
echo "  python evaluation/self_check.py --all"
