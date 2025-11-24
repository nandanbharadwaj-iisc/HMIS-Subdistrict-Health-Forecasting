HMIS Subdistrict Health Trend Forecasting
=========================================

Project Overview
----------------
This project builds an end–to–end forecasting pipeline and dashboard for
HMIS (Health Management Information System) indicators at **subdistrict
level** in India.

The workflow has two main parts:

1. Jupyter / PySpark pipeline (offline model building)
   - Reads the raw HMIS dataset
     (`data/major-health-indicators-subdistrict-level.csv`).
   - Performs feature engineering (time-index, lags, rolling means).
   - Trains multiple regression models per indicator (e.g. Linear
     Regression, Random Forest, Gradient Boosted Trees).
   - Evaluates model performance using RMSE, MAE, MAPE, R².
   - Exports model performance and time-series forecasts as CSVs for the
     dashboard.

2. Streamlit dashboard (interactive exploration)
   - Uses the exported CSVs in `streamlit_app/data/`:
     - `metrics_v6.csv` – per-target, per-model metrics.
     - `best_model_per_target_v6.csv` – best model summary.
     - `forecast_timeseries.csv` – actual vs forecast values over time
       (with state / district / subdistrict information).
     - `india_states_geo.json` – GeoJSON boundaries for Indian states.
   - Provides:
     - National overview of actual vs forecast trend.
     - India state-level spatial heatmap.
     - District / Subdistrict explorer for drilling into specific
       locations.
     - Model performance comparison across algorithms.

Technology Stack
----------------
- Python 3.11.14
- PySpark (for feature engineering & model training)
- Pandas
- Plotly
- Streamlit
- Jupyter / IPython kernel
- (Optional) Google Colab for running the notebook in the cloud

Repository Layout
-----------------
HMIS-Subdistrict-Health-Forecasting/
├─ README.md                     (optional, GitHub rendered)
├─ readme.txt                    (this file)
├─ setup.sh                      (venv + requirements + Jupyter launcher)
├─ requirements.txt              (Python dependencies)
├─ HMIS_SubDistrict_Health_Trend_Forecasting.ipynb
├─ output_dir/                   (Output directory for Jupyter notebook output CSVs)
├─ data/
│   └─ major-health-indicators-subdistrict-level.csv
└─ streamlit_app/
   ├─ app.py
   └─ data/
       ├─ best_model_per_target_v6.csv
       ├─ forecast_timeseries.csv
       ├─ india_states_geo.json
       └─ metrics_v6.csv

Data Flow
---------
1. Input: `data/major-health-indicators-subdistrict-level.csv`
   - Raw HMIS indicators at subdistrict–month or subdistrict–quarter
     level.

2. Notebook:
   - `HMIS_SubDistrict_Health_Trend_Forecasting.ipynb`
   - Outputs (paths configurable inside the notebook):
     - Metrics CSV (merged later into `metrics_v6.csv`).
     - Best model per target CSV (`best_model_per_target_v6.csv`).
     - Forecast time series CSV (`forecast_timeseries.csv`).

3. Dashboard:
   - `streamlit_app/app.py` reads the CSVs from `streamlit_app/data/`
     and exposes:
     - Overview tab (national trend + map).
     - District Explorer tab.
     - Model Performance tab.

Local Setup (Linux / macOS, Python 3.11.14)
-------------------------------------------
Pre-requisites:
- Git
- Python 3.11.14 available as `python3.11` (e.g. via pyenv, Homebrew, or system package manager)
- Node is NOT required; Streamlit runs purely in Python.

Steps:

1. Clone the repository:

   git clone <your_repo_url>.git
   cd HMIS-Subdistrict-Health-Forecasting

2. Make the setup script executable:

   chmod +x setup.sh

3. Run the setup script (creates venv, installs requirements, starts Jupyter):

   ./setup.sh

   The script will:
   - Create a `.venv` virtual environment using Python 3.11.
   - Install all Python dependencies from `requirements.txt`.
   - Install an IPython kernel named `hmis_py311`.
   - Launch Jupyter Lab in the project directory.

4. (Optional) If you prefer manual steps instead of `setup.sh`:

   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m ipykernel install --user --name "hmis_py311" --display-name "Python 3.11 (HMIS)"
   jupyter lab

Running the PySpark Notebook Locally
------------------------------------
1. Ensure the virtual environment is active:

   source .venv/bin/activate

2. Start Jupyter (if not already started by setup.sh):

   jupyter lab

3. Open:

   HMIS_SubDistrict_Health_Trend_Forecasting.ipynb

4. Configure input / output paths inside the notebook if needed:
   - Input: `data/major-health-indicators-subdistrict-level.csv`
   - Output: CSVs that will later be copied to `streamlit_app/data/`:
     - `metrics_v6.csv`
     - `best_model_per_target_v6.csv`
     - `forecast_timeseries.csv`

5. Run all cells. After completion, make sure:
   - The metrics and forecast CSVs are present (either in `streamlit_app/data/`
     or moved there manually).
   - No errors are reported in model training / evaluation cells.

Running the Streamlit Dashboard Locally
---------------------------------------
1. Ensure the virtual environment is active:

   source .venv/bin/activate

2. Verify the dashboard data directory exists in `streamlit_app/data/` with `india_states_geo.json`:

   - best_model_per_target_v6.csv
   - forecast_timeseries.csv
   - metrics_v6.csv
   Streamlit backend will copy the above files from Jupuyter output directory `HMIS-Subdistrict-Health-Forecasting/output_dir` to `streamlit_app/data/`

3. Launch the Streamlit app:

   cd streamlit_app
   streamlit run app.py

4. Open the URL shown in the terminal (typically http://localhost:8501)
   in your browser.

5. Use the sidebar filters to:
   - Select a Target Indicator.
   - Select a Model for diagnostics.
   - Choose how the map aggregates values (latest quarter vs all quarters).

Running the Notebook in Google Colab
------------------------------------
You can also run the PySpark notebook on Google Colab if you do not
want to install Spark locally.

1. Upload the notebook and dataset:
   - Go to https://colab.research.google.com
   - Click "Upload" and select `HMIS_SubDistrict_Health_Trend_Forecasting.ipynb`.
   - Either:
     - Upload `major-health-indicators-subdistrict-level.csv` via the
       Colab file browser, or
     - Put it in your Google Drive and mount Drive inside Colab.

2. Install dependencies at the top of the notebook (add a new cell):

   !pip install pyspark==3.5.0 pandas plotly

   # Optional: if you use any other libraries (e.g., scikit-learn):
   !pip install scikit-learn

3. (Optional) Mount Google Drive (if your data is in Drive):

   from google.colab import drive
   drive.mount("/content/drive")

   # Example: change to a project folder in Drive
   %cd /content/drive/MyDrive/HMIS-Subdistrict-Health-Forecasting

4. Adjust file paths in the notebook:
   - Make sure the path to the CSV uses the Colab location, e.g.:

     DATA_PATH = "/content/major-health-indicators-subdistrict-level.csv"
     # or, if using Drive:
     # DATA_PATH = "/content/drive/MyDrive/.../major-health-indicators-subdistrict-level.csv"

5. Run all cells:
   - Colab will start a Spark session using the installed PySpark.
   - Evaluate metrics and export outputs as CSVs.

6. Download outputs for the dashboard:
   - After the notebook exports `metrics_v6.csv`, `best_model_per_target_v6.csv`
     and `forecast_timeseries.csv`, download them from Colab.
   - Place them into `streamlit_app/data/` in your local clone.
   - Then run the Streamlit app locally as described above.

Extending the Project
---------------------
Some ideas for future versions:
- Add anomaly detection dashboard (e.g., highlight districts with
  unusually high forecast errors).
- Add more models (e.g., XGBoost / LightGBM via their Python APIs).
- Add indicator-level configuration (e.g., choose lag depth or window
  size per indicator).
- Expose model hyperparameters as sliders in the Streamlit UI and
  re-train lightweight models on the fly.
