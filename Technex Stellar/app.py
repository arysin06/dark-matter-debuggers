import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os

# --- Configuration ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
CLASS_PIPELINE_FILES = ['pipeline_A_v2.pkl', 'pipeline_A_v2.joblib']
REG_PIPELINE_FILES = ['pipeline_B_v2.pkl', 'pipeline_B_v2.joblib']
FEATURE_A_FILE = 'features_A_selected.pkl'
FEATURE_B_FILE = 'features_B_selected.pkl'
IMPUTER_A_FILE = 'imputer_A.pkl'
IMPUTER_B_FILE = 'imputer_B.pkl'

# Required input fields (26 attributes from CSV header)
REQUIRED_FIELDS = [
    'kepid','koi_disposition','koi_period','koi_duration','koi_depth','koi_impact',
    'koi_model_snr','koi_num_transits','koi_ror','koi_prad','st_teff','st_logg',
    'st_met','st_mass','st_radius','st_dens','teff_err1','teff_err2','logg_err1',
    'logg_err2','feh_err1','feh_err2','mass_err1','mass_err2','radius_err1','radius_err2'
]

# --- Helper: robust model loader with fallback paths ---

def load_model_file(filenames):
    for fname in filenames:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            return joblib.load(path)
        # fallback to root
        path_root = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(path_root):
            return joblib.load(path_root)
    raise FileNotFoundError(f"None of {filenames} found in {MODEL_DIR} or project root")


# --- Feature engineering function (same logic used during training) ---
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Stellar density (approx mass / radius^3)
    if 'st_mass' in df.columns and 'st_radius' in df.columns:
        df['stellar_density'] = df['st_mass'] / (df['st_radius'] ** 3 + 1e-9)

    # Relative errors
    error_mappings = {
        'st_teff': ('teff_err1', 'teff_err2'),
        'st_logg': ('logg_err1', 'logg_err2'),
        'st_met': ('feh_err1', 'feh_err2'),
        'st_mass': ('mass_err1', 'mass_err2'),
        'st_radius': ('radius_err1', 'radius_err2')
    }
    for base_col, (err1_col, err2_col) in error_mappings.items():
        if base_col in df.columns and err1_col in df.columns and err2_col in df.columns:
            df[f'{base_col}_rel_err'] = (
                (df[err1_col].abs() + df[err2_col].abs()) / 2.0 / (df[base_col].abs() + 1e-9)
            )

    # Uncertainty magnitudes
    uncertainty_features = {
        'teff': ('teff_err1', 'teff_err2'),
        'logg': ('logg_err1', 'logg_err2'),
        'feh': ('feh_err1', 'feh_err2'),
        'mass': ('mass_err1', 'mass_err2'),
        'radius': ('radius_err1', 'radius_err2')
    }
    for base, (err1, err2) in uncertainty_features.items():
        if err1 in df.columns and err2 in df.columns:
            df[f'{base}_uncertainty'] = (df[err1].abs() + df[err2].abs()) / 2.0

    # domain-specific features
    if 'koi_depth' in df.columns and 'koi_period' in df.columns:
        df['depth_per_period'] = df['koi_depth'] / (df['koi_period'] + 1e-9)
    if 'koi_model_snr' in df.columns and 'koi_num_transits' in df.columns:
        df['snr_per_transit'] = df['koi_model_snr'] / (df['koi_num_transits'] + 1e-9)
    if 'koi_impact' in df.columns and 'koi_ror' in df.columns:
        df['impact_ror_interaction'] = df['koi_impact'] * df['koi_ror']

    return df


# --- Load artifacts at startup ---
try:
    CLASS_PIPELINE = load_model_file(CLASS_PIPELINE_FILES)
    REG_PIPELINE = load_model_file(REG_PIPELINE_FILES)
    FEATURES_A = joblib.load(os.path.join(MODEL_DIR, FEATURE_A_FILE)) if os.path.exists(os.path.join(MODEL_DIR, FEATURE_A_FILE)) else joblib.load(FEATURE_A_FILE)
    FEATURES_B = joblib.load(os.path.join(MODEL_DIR, FEATURE_B_FILE)) if os.path.exists(os.path.join(MODEL_DIR, FEATURE_B_FILE)) else joblib.load(FEATURE_B_FILE)
    IMPUTER_A = joblib.load(os.path.join(MODEL_DIR, IMPUTER_A_FILE)) if os.path.exists(os.path.join(MODEL_DIR, IMPUTER_A_FILE)) else joblib.load(IMPUTER_A_FILE)
    IMPUTER_B = joblib.load(os.path.join(MODEL_DIR, IMPUTER_B_FILE)) if os.path.exists(os.path.join(MODEL_DIR, IMPUTER_B_FILE)) else joblib.load(IMPUTER_B_FILE)
except Exception as e:
    # If artifact loading fails, raise so server won't start silently broken
    raise RuntimeError(f"Failed to load model artifacts: {e}")

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'ok',
        'service': 'Stellar Analytics API',
        'endpoints': {
            'predict': {
                'method': 'POST',
                'description': 'Send a JSON object with the 26 required fields to get predictions'
            }
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/favicon.ico')
def favicon():
    return ('', 204)


# --- Request validation ---
def validate_payload(payload: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return False, f"Missing required fields: {missing}"
    # Try cast numeric fields (all except koi_disposition) to float
    numeric_fields = [f for f in REQUIRED_FIELDS if f != 'koi_disposition']
    for nf in numeric_fields:
        try:
            _ = float(payload[nf]) if payload[nf] is not None else float('nan')
        except Exception:
            return False, f"Field '{nf}' must be numeric or parseable as float"
    return True, None


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({'status': 'error', 'message': 'Invalid JSON payload'}), 400

        # If the client sends a list, take first element (we expect single record)
        if isinstance(payload, list):
            if len(payload) == 0:
                return jsonify({'status': 'error', 'message': 'Empty list provided'}), 400
            payload = payload[0]

        ok, err = validate_payload(payload)
        if not ok:
            return jsonify({'status': 'error', 'message': err}), 400

        # Build DataFrame with required fields in arbitrary order
        data = {k: payload.get(k, None) for k in REQUIRED_FIELDS}

        # Cast numeric fields
        for k in data:
            if k == 'koi_disposition':
                continue
            val = data[k]
            try:
                data[k] = float(val) if val is not None and val != '' else np.nan
            except Exception:
                data[k] = np.nan

        df = pd.DataFrame([data])

        # Feature engineering
        df = feature_engineer(df)

        # Prepare features for classification
        X_A = df.copy()
        # Ensure columns exist for all FEATURES_A (missing engineered cols filled with NaN)
        for col in FEATURES_A:
            if col not in X_A.columns:
                X_A[col] = np.nan
        X_A = X_A[FEATURES_A]
        # Impute using simple median imputation for single-record inference
        X_A = X_A.fillna(X_A.median())
        X_A_imputed = X_A.copy()

        # Prepare features for regression
        X_B = df.copy()
        for col in FEATURES_B:
            if col not in X_B.columns:
                X_B[col] = np.nan
        X_B = X_B[FEATURES_B]
        # Impute for regression using simple median imputation for single-record inference
        X_B = X_B.fillna(X_B.median())
        X_B_imputed = X_B.copy()

        # Run models
        # Pass numpy arrays to avoid pandas feature-name strict checks
        disp_proba = CLASS_PIPELINE.predict_proba(X_A_imputed.values)[:, 1][0]
        disp_label = CLASS_PIPELINE.predict(X_A_imputed.values)[0]
        disp_label_str = 'CONFIRMED' if int(disp_label) == 1 else 'FALSE POSITIVE'

        # Regression: pipeline predicts log1p target — invert with expm1
        pred_log = REG_PIPELINE.predict(X_B_imputed.values)[0]
        try:
            pred_radius = float(np.expm1(pred_log))
        except Exception:
            # If regressor was trained on raw target, fallback
            pred_radius = float(pred_log)

        result = {
            'status': 'success',
            'data': {
                'disposition_prediction': disp_label_str,
                'disposition_probability': float(np.round(disp_proba, 6)),
                'predicted_radius_earth': float(np.round(pred_radius, 6))
            }
        }
        return jsonify(result), 200

    except Exception as e:
        # Log full traceback to console for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Inference failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Run development server
    app.run(host='0.0.0.0', port=5000, debug=False)
