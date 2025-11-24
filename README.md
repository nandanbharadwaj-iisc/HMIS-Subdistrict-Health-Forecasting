# HMIS Subdistrict Health Trend Forecasting

End–to–end forecasting pipeline and dashboard for **HMIS (Health Management Information System)** indicators at **subdistrict level** in India.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Data Flow & Architecture](#data-flow--architecture)
- [Local Setup (Linux / macOS)](#local-setup-linux--macos)
- [Running the PySpark Notebook](#running-the-pyspark-notebook)
- [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
- [Running the Notebook in Google Colab](#running-the-notebook-in-google-colab)
- [Extending the Project](#extending-the-project)

---

## Project Overview

This project builds an **end–to–end forecasting pipeline** and an **interactive dashboard** for HMIS indicators at the **subdistrict level** in India.

The workflow has two main parts:

1. **Jupyter / PySpark pipeline (offline model building)**
   - Reads the raw HMIS dataset:  
     `data/major-health-indicators-subdistrict-level.csv`
   - Performs time series feature engineering:
     - Time index
     - Lag features
     - Rolling means
   - Trains multiple regression models per indicator:
     - Linear Regression
     - Random Forest
     - Gradient Boosted Trees
   - Evaluates model performance using:
     - RMSE, MAE, MAPE, R²
   - Exports model performance and time–series forecasts as CSVs for the dashboard.

2. **Streamlit dashboard (interactive exploration)**
   - Uses the exported CSVs in `streamlit_app/data/`:
     - `metrics_v6.csv` – per-target, per-model metrics  
     - `best_model_per_target_v6.csv` – best model summary  
     - `forecast_timeseries.csv` – actual vs forecast values over time  
     - `india_states_geo.json` – GeoJSON boundaries for Indian states
   - Provides:
     - National overview of actual vs forecast trend
     - India state-level spatial heatmap
     - District / Subdistrict explorer
     - Model performance comparison across algorithms

---

## Features

- End–to–end pipeline from raw HMIS CSV → trained models → forecast CSVs
- Multiple ML models per indicator with metric comparison
- Spatial heatmap of Indian states using GeoJSON
- Drill-down explorer:
  - State → District → Subdistrict
- Model performance dashboard:
  - RMSE, MAE, MAPE, R² per model & indicator
- Optional Google Colab support (no local Spark installation required)

---

## Technology Stack

- Python **3.11.14**
- **PySpark** – feature engineering & model training
- **Pandas**
- **Plotly** – interactive charts & maps
- **Streamlit** – dashboard UI
- **Jupyter / IPython** – notebook workflow
- *(Optional)* Google Colab – cloud execution

---

## Repository Structure

```text
HMIS-Subdistrict-Health-Forecasting/
├─ README.md                     # GitHub-friendly README (this file)
├─ readme.txt                    # Original text version
├─ setup.sh                      # venv + requirements + Jupyter launcher
├─ requirements.txt              # Python dependencies
├─ HMIS_SubDistrict_Health_Trend_Forecasting.ipynb
├─ output_dir/                   # Output CSVs from Jupyter pipeline
├─ data/
│   └─ major-health-indicators-subdistrict-level.csv
└─ streamlit_app/
   ├─ app.py
   └─ data/
       ├─ best_model_per_target_v6.csv
       ├─ forecast_timeseries.csv
       ├─ india_states_geo.json
       └─ metrics_v6.csv
```

---

## Data Flow & Architecture

1. **Input (Raw Data)**
   - `data/major-health-indicators-subdistrict-level.csv`
   - HMIS indicators at subdistrict–month or subdistrict–quarter level.

2. **Offline Modeling (Notebook)**
   - `HMIS_SubDistrict_Health_Trend_Forecasting.ipynb`
   - Performs:
     - Data cleaning & aggregation
     - Feature engineering (lags, rolling windows)
     - Model training & evaluation
   - Outputs CSVs (paths configurable inside the notebook):
     - `metrics_v6.csv` – per-target, per-model metrics
     - `best_model_per_target_v6.csv` – best model per indicator
     - `forecast_timeseries.csv` – actual vs forecast values with
       state/district/subdistrict/time

3. **Dashboard (Streamlit)**
   - `streamlit_app/app.py` reads CSVs from `streamlit_app/data/`
   - Exposes:
     - **Overview tab** – national trend + map
     - **District Explorer tab** – drill-down by geography
     - **Model Performance tab** – compare models & metrics

> Note: The Jupyter notebook writes outputs to `output_dir/`.  
> The Streamlit backend (or a helper script) copies these into `streamlit_app/data/`.

---

## Local Setup (Linux / macOS)

### Prerequisites

- Git
- Python **3.11.14** available as `python3.11`  
  (via pyenv, Homebrew, or system package manager)
- Node.js **not required** (Streamlit runs purely in Python)

### 1. Clone the repository

```bash
git clone <your_repo_url>.git
cd HMIS-Subdistrict-Health-Forecasting
```

### 2. Make the setup script executable

```bash
chmod +x setup.sh
```

### 3. Run the setup script

```bash
./setup.sh
```

The script will:

- Create a `.venv` virtual environment using Python 3.11
- Install all dependencies from `requirements.txt`
- Install an IPython kernel named `hmis_py311`
- Launch Jupyter Lab in the project directory

---

### Manual Setup (Alternative to `setup.sh`)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user   --name "hmis_py311"   --display-name "Python 3.11 (HMIS)"

jupyter lab
```

---

## Running the PySpark Notebook

1. **Activate the virtual environment**

```bash
source .venv/bin/activate
```

2. **Start Jupyter Lab**

```bash
jupyter lab
```

3. **Open the notebook**

- Open `HMIS_SubDistrict_Health_Trend_Forecasting.ipynb` in Jupyter.

4. **Check / configure paths inside the notebook**

- Input:
  ```python
  DATA_PATH = "data/major-health-indicators-subdistrict-level.csv"
  ```
- Output CSVs (later consumed by Streamlit):
  - `metrics_v6.csv`
  - `best_model_per_target_v6.csv`
  - `forecast_timeseries.csv`

5. **Run all cells**

- Ensure that:
  - Model training & evaluation cells run without errors.
  - Output CSVs are generated (in `output_dir/` or directly in `streamlit_app/data/`).

6. **Copy outputs for the dashboard (if needed)**

If the notebook writes to `output_dir/`, copy the files to `streamlit_app/data/`:

```bash
cp output_dir/metrics_v6.csv streamlit_app/data/
cp output_dir/best_model_per_target_v6.csv streamlit_app/data/
cp output_dir/forecast_timeseries.csv streamlit_app/data/
```

---

## Running the Streamlit Dashboard

1. **Activate the virtual environment**

```bash
source .venv/bin/activate
```

2. **Verify dashboard data files**

In `streamlit_app/data/`, ensure you have:

- `best_model_per_target_v6.csv`
- `forecast_timeseries.csv`
- `metrics_v6.csv`
- `india_states_geo.json`

> Note: The Streamlit backend can also copy the CSVs from  
> `HMIS-Subdistrict-Health-Forecasting/output_dir` to `streamlit_app/data/`.

3. **Launch Streamlit app**

```bash
cd streamlit_app
streamlit run app.py
```

4. **Open the app in your browser**

- Default URL (shown in terminal):  
  `http://localhost:8501`

5. **Use the sidebar controls**

- Select:
  - Target indicator
  - Model for diagnostics
  - Aggregation mode (e.g., latest quarter vs all quarters)

---

## Running the Notebook in Google Colab

You can run the PySpark notebook on **Google Colab** if you prefer not to install Spark locally.

1. **Upload the notebook and dataset**

- Go to [Google Colab](https://colab.research.google.com)
- Click **Upload** and select:
  - `HMIS_SubDistrict_Health_Trend_Forecasting.ipynb`
- For the dataset:
  - Upload `major-health-indicators-subdistrict-level.csv` via Colab file browser, **or**
  - Place it in Google Drive and mount Drive in Colab

2. **Install dependencies in Colab**

At the top of the notebook, add:

```python
!pip install pyspark==3.5.0 pandas plotly
# Optional: if you use additional libraries
!pip install scikit-learn
```

3. **(Optional) Mount Google Drive**

```python
from google.colab import drive
drive.mount("/content/drive")

# Example: change to a project folder in Drive
%cd /content/drive/MyDrive/HMIS-Subdistrict-Health-Forecasting
```

4. **Adjust file paths**

Example:

```python
DATA_PATH = "/content/major-health-indicators-subdistrict-level.csv"
# or, if using Drive:
# DATA_PATH = "/content/drive/MyDrive/.../major-health-indicators-subdistrict-level.csv"
```

5. **Run all cells**

- Colab will start a Spark session using PySpark.
- Metrics and forecast CSVs will be generated.

6. **Download outputs for the dashboard**

- Download:
  - `metrics_v6.csv`
  - `best_model_per_target_v6.csv`
  - `forecast_timeseries.csv`
- Place them into `streamlit_app/data/` in your local clone.
- Run the Streamlit app locally (as described above).

---

> 💡 Contributions, issues, and feature requests are welcome.  
> Feel free to open an issue or submit a pull request!
