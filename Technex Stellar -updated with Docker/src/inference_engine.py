import os

# compute base directory of project (one level above src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# default location for models lives under the project root
# allow override via MODEL_DIR environment variable (useful in containers)
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(BASE_DIR, 'models'))


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # copied from app.py; keeps pipelines consistent with training
    if 'st_mass' in df.columns and 'st_radius' in df.columns:
        df['stellar_density'] = df['st_mass'] / (df['st_radius'] ** 3 + 1e-9)

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

    if 'koi_depth' in df.columns and 'koi_period' in df.columns:
        df['depth_per_period'] = df['koi_depth'] / (df['koi_period'] + 1e-9)
    if 'koi_model_snr' in df.columns and 'koi_num_transits' in df.columns:
        df['snr_per_transit'] = df['koi_model_snr'] / (df['koi_num_transits'] + 1e-9)
    if 'koi_impact' in df.columns and 'koi_ror' in df.columns:
        df['impact_ror_interaction'] = df['koi_impact'] * df['koi_ror']

    return df


class InferenceEngine:
    """Encapsulates model loading and prediction logic.

    The engine reads pipelines, imputers and feature lists from disk on
    initialization.  It builds a dynamic required-feature set so that new
    attributes may be added in a later round without modifying validation
    code.

    If model artifacts are missing the engine will still initialize but
    pipelines will remain ``None``.  The Flask health endpoint checks this
    and can report degraded status without crashing the service.
    """

    def __init__(self):
        self.classify_pipeline = None
        self.regress_pipeline = None
        self.imputer_A = None
        self.imputer_B = None
        self.features_A = []
        self.features_B = []

        # report what MODEL_DIR looks like when first constructed
        print(f"[InferenceEngine] using MODEL_DIR={MODEL_DIR}")
        try:
            print(f"[InferenceEngine] contents: {os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else 'not a directory'}")
        except Exception as _:
            print(f"[InferenceEngine] could not list MODEL_DIR contents")

        try:
            self._load_artifacts()
        except Exception as e:
            # log warning but don't re-raise to avoid import-time crash
            print(f"[InferenceEngine] WARNING: failed to load artifacts: {e}")
            # leave pipelines as None; health check will catch this

    def _load_artifacts(self):
        def _load(fname_list):
            for fname in fname_list:
                path = os.path.join(MODEL_DIR, fname)
                if os.path.exists(path):
                    return joblib.load(path)
            # return None instead of raising so caller can handle gracefully
            return None

        # pipelines may have either .joblib or .pkl extension
        self.classify_pipeline = _load(['pipeline_A_v2.joblib', 'pipeline_A_v2.pkl'])
        if self.classify_pipeline is None:
            print(f"[InferenceEngine] classify pipeline not found in {MODEL_DIR}")

        self.regress_pipeline = _load(['pipeline_B_v2.joblib', 'pipeline_B_v2.pkl'])
        if self.regress_pipeline is None:
            print(f"[InferenceEngine] regress pipeline not found in {MODEL_DIR}")

        # imputers (used because pipelines above don't perform imputation)
        try:
            self.imputer_A = joblib.load(os.path.join(MODEL_DIR, 'imputer_A.pkl'))
        except Exception as e:
            print(f"[InferenceEngine] warning loading imputer_A: {e}")
            self.imputer_A = None
        try:
            self.imputer_B = joblib.load(os.path.join(MODEL_DIR, 'imputer_B.pkl'))
        except Exception as e:
            print(f"[InferenceEngine] warning loading imputer_B: {e}")
            self.imputer_B = None

        # feature lists – if missing we can leave empty lists
        try:
            fA = joblib.load(os.path.join(MODEL_DIR, 'features_A_selected.pkl'))
            self.features_A = list(fA)
        except Exception as e:
            print(f"[InferenceEngine] warning loading features_A_selected: {e}")
            self.features_A = []
        try:
            fB = joblib.load(os.path.join(MODEL_DIR, 'features_B_selected.pkl'))
            self.features_B = list(fB)
        except Exception as e:
            print(f"[InferenceEngine] warning loading features_B_selected: {e}")
            self.features_B = []

    def required_fields(self):
        # include koi_disposition separately since it's not part of the
        # numerical feature lists but is needed for validation and metadata.
        return set(self.features_A) | set(self.features_B) | {'koi_disposition'}

    def validate_payload(self, payload: dict):
        # original required_fields were derived from post-engineering
        # feature lists; some of those are constructed features that the
        # frontend won't supply.  prediction() can handle NaN values, so we
        # no longer treat missing engineered fields as fatal.
        missing = [f for f in self.required_fields() if f not in payload]
        if missing:
            # log for debugging but proceed
            print(f"[InferenceEngine] warning - missing fields in payload: {missing}")
            # do not return False; we'll fill NaNs later in predict

        # numeric validation for every field except koi_disposition
        for k, v in payload.items():
            if k == 'koi_disposition':
                continue
            try:
                # allow None / empty strings (will be imputed later)
                if v is not None and v != '':
                    float(v)
            except Exception:
                return False, f"Field '{k}' must be numeric or parseable as float"
        return True, None

    def predict(self, payload: dict):
        # assume payload already validated
        data = {k: payload.get(k, None) for k in self.required_fields()}

        # cast numeric fields
        for k in data:
            if k == 'koi_disposition':
                continue
            val = data[k]
            try:
                data[k] = float(val) if val is not None and val != '' else np.nan
            except Exception:
                data[k] = np.nan

        df = pd.DataFrame([data])

        # Impute using all features that will be used (before feature engineering)
        # The imputers were trained on the full 27-feature set from features_B
        if self.imputer_B is not None and self.features_B:
            try:
                # Build array with just the raw features in the order expected by imputer
                X_raw = np.full((1, len(self.features_B)), np.nan)
                for i, feat in enumerate(self.features_B):
                    if feat in df.columns:
                        X_raw[0, i] = df[feat].iloc[0]
                    # else: will be NaN, gets imputed
                
                # Apply imputation
                X_raw_imputed = self.imputer_B.transform(X_raw)
                
                # Put imputed values back into dataframe
                for i, feat in enumerate(self.features_B):
                    df[feat] = X_raw_imputed[0, i]
            except Exception as e:
                print(f"[InferenceEngine] warning - imputation failed: {e}")

        # apply same feature engineering used during training
        df = feature_engineer(df)

        # prepare arrays for pipelines (now that we have all engineered features and imputed values)
        X_A = df[self.features_A].values if self.features_A else np.empty((1, 0))
        X_B = df[self.features_B].values if self.features_B else np.empty((1, 0))

        # run classifiers/regressors
        disp_proba = self.classify_pipeline.predict_proba(X_A)[:, 1][0]
        disp_label = self.classify_pipeline.predict(X_A)[0]
        disp_label_str = 'CONFIRMED' if int(disp_label) == 1 else 'FALSE POSITIVE'

        pred_log = self.regress_pipeline.predict(X_B)[0]
        try:
            pred_radius = float(np.expm1(pred_log))
        except Exception:
            pred_radius = float(pred_log)

        return {
            'disposition_prediction': disp_label_str,
            'disposition_probability': float(np.round(disp_proba, 6)),
            'predicted_radius_earth': float(np.round(pred_radius, 6))
        }


# singleton instance for the Flask app to reuse
engine = InferenceEngine()
