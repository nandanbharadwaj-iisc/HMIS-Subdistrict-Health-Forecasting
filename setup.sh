#!/usr/bin/env bash
#
# setup.sh
#
# One-shot environment setup script for:
# - Python 3.11 virtual environment
# - requirements.txt installation
# - Jupyter kernel registration
# - Jupyter Lab launch
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Optionally override the Python binary via:
#   PYTHON_BIN=/path/to/python3.11 ./setup.sh
#

set -euo pipefail

# Resolve project root (directory containing this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Python 3.11 binary (can be overridden from the environment)
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "==================================================="
echo " HMIS Subdistrict Health Forecasting - setup"
echo " Project root : $PROJECT_ROOT"
echo " Python binary: $PYTHON_BIN"
echo "==================================================="

# Check that python exists and is 3.11.x
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' not found on PATH."
  echo "Please install Python 3.11.14 (e.g., via pyenv or system packages) and try again."
  exit 1
fi

"$PYTHON_BIN" -V

# Create virtual environment if not present
if [ ! -d ".venv" ]; then
  echo "[INFO] Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "[INFO] Virtual environment .venv already exists. Reusing."
fi

# Activate venv
# shellcheck source=/dev/null
source .venv/bin/activate

echo "[INFO] Using Python from venv: $(which python)"
python -V

# Upgrade pip
echo "[INFO] Upgrading pip ..."
python -m pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
  echo "[INFO] Installing Python dependencies from requirements.txt ..."
  python -m pip install -r requirements.txt
else
  echo "[WARN] requirements.txt not found in project root. Skipping requirements install."
fi

# Ensure Jupyter + kernel are available
echo "[INFO] Installing Jupyter + IPython kernel ..."
python -m pip install jupyter ipykernel

# Register a named kernel for this env
KERNEL_NAME="hmis_py311"
echo "[INFO] Registering Jupyter kernel '$KERNEL_NAME' ..."
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python 3.11 (HMIS)"

echo "==================================================="
echo " Setup complete."
echo " Virtual env : .venv"
echo " Kernel name : $KERNEL_NAME"
echo "==================================================="
echo ""
echo "To activate the environment manually:"
echo "  source .venv/bin/activate"
echo ""
echo "To run Jupyter Lab manually (after activation):"
echo "  jupyter lab"
echo ""
echo "To run the Streamlit dashboard (after activation):"
echo "  cd streamlit_app"
echo "  streamlit run app.py"
echo ""
echo "[INFO] Launching Jupyter Lab now ..."
jupyter lab