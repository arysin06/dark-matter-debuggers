import numpy as np
import pandas as pd
import os  # <--- Add this import
from flask import Flask, request, jsonify

# Define the absolute path to your models folder
# This ensures Docker finds the folder at /app/models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models') 

# Now import the engine
from src.inference_engine import engine
# --- configuration / metadata
# The engine dynamically computes the set of required fields based on
# the feature lists that were serialized during training. It also handles
# loading models from the directory specified by MODEL_DIR env var.

REQUIRED_FIELDS = list(engine.required_fields())




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
    # entry point used by front-end to check service status; always return 200
    status = 'healthy'
    details = {}
    # include model directory for debugging
    mdl = os.getenv('MODEL_DIR', os.path.join(os.path.dirname(__file__), 'models'))
    try:
        contents = os.listdir(mdl) if os.path.isdir(mdl) else []
        details['model_dir'] = mdl
        details['files'] = contents
    except Exception as e:
        details['model_dir_error'] = str(e)

    try:
        if engine.classify_pipeline is None:
            status = 'degraded'
            details['classify_pipeline'] = 'missing'
        else:
            details['classify_pipeline'] = 'loaded'
        if engine.regress_pipeline is None:
            status = 'degraded'
            details['regress_pipeline'] = 'missing'
        else:
            details['regress_pipeline'] = 'loaded'
    except Exception as e:
        status = 'degraded'
        details['error'] = str(e)
    return jsonify({'status': status, **details}), 200


@app.route('/favicon.ico')
def favicon():
    return ('', 204)


# --- Request validation ---
# delegate to engine which computes required_fields dynamically

def validate_payload(payload: dict):
    return engine.validate_payload(payload)


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
            print(f"[app] payload validation failed: {err}, payload={payload}")
            return jsonify({'status': 'error', 'message': err}), 400

        # delegate to engine for prediction; engine handles casting, imputation and
        # feature engineering internally
        result = engine.predict(payload)
        return jsonify({'status': 'success', 'data': result}), 200

    except Exception as e:
        # Log full traceback to console for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Inference failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Run development server
    app.run(host='0.0.0.0', port=5000, debug=False)
