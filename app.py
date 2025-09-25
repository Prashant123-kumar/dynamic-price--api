# app.py
import os
import traceback
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS   # <- added

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # <- allow all origins for dev

# -------------------------
# Configuration
# -------------------------
# Dynamic price multiplier
ALPHA = float(os.getenv("DYNAMIC_PRICE_ALPHA", "0.05"))

# Database URL:
DATABASE_URL = os.getenv("DATABASE_URL", None)
if DATABASE_URL:
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictions.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# prefer environment variable, else try local dev path, else docker path
DEFAULT_LOCAL = os.path.join(os.getcwd(), "models", "dynamic_pricing_model.pkl")
DOCKER_DEFAULT = "/app/models/dynamic_pricing_model.pkl"
MODEL_PATH = os.getenv("DYNAMIC_MODEL_PATH", None)
if not MODEL_PATH:
    if os.path.exists(DEFAULT_LOCAL):
        MODEL_PATH = DEFAULT_LOCAL
    else:
        MODEL_PATH = DOCKER_DEFAULT

# -------------------------
# Initialize DB
# -------------------------
db = SQLAlchemy(app)

class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(100))
    predicted_sales = db.Column(db.Float)
    dynamic_price = db.Column(db.Float)
    actual_sales = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

with app.app_context():
    db.create_all()

# -------------------------
# Load model
# -------------------------
model = None
load_errors = []
try:
    model = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded from %s", MODEL_PATH)
    app.logger.info("Model type: %s", type(model))
    if hasattr(model, "feature_names_in_"):
        app.logger.info("Model features: %s", list(model.feature_names_in_))
    if hasattr(model, "named_steps"):
        app.logger.info("Model steps: %s", list(model.named_steps.keys()))
except Exception as e:
    load_errors.append(str(e))
    app.logger.error("Failed to load model: %s", e)

# -------------------------
# Helpers
# -------------------------
def _prepare_dataframe_from_dicts(dicts, expected_features):
    """
    Create a DataFrame from the list of input dicts, drop target column if present,
    validate presence of expected features (if provided), ignore extras.
    """
    df = pd.DataFrame(dicts)

    # Drop target if included by mistake
    if "Item_Outlet_Sales" in df.columns:
        app.logger.warning("Dropping 'Item_Outlet_Sales' from input data (target must not be provided).")
        df = df.drop(columns=["Item_Outlet_Sales"])

    # Align to expected features if model provides them
    if expected_features is not None:
        expected = list(expected_features)
        # ensure target not in expected features used for inference
        if "Item_Outlet_Sales" in expected:
            expected = [c for c in expected if c != "Item_Outlet_Sales"]

        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]

        if missing:
            raise ValueError(f"Missing required features: {missing}")
        if extra:
            app.logger.warning("Ignoring extra features: %s", extra)

        # Reorder and select only expected columns
        df = df[expected]

    return df


def _make_response_and_log(df, preds, original_dicts=None):
    """
    Build response rows (echo input + predicted_sales + dynamic_price), log each prediction to DB.
    original_dicts: list of original payload dicts (used to fetch product_id even if dropped).
    """
    pred_vals = [float(x) for x in np.asarray(preds).reshape(-1).tolist()]

    # Compute dynamic prices
    dynamic_prices = []
    if "Item_MRP" in df.columns:
        for i, p in enumerate(pred_vals):
            try:
                mrp = float(df["Item_MRP"].iloc[i])
                dynamic_prices.append(round(mrp + ALPHA * p, 4))
            except Exception:
                dynamic_prices.append(None)
    else:
        dynamic_prices = [None] * len(pred_vals)

    response_rows = []
    for i in range(len(pred_vals)):
        # base row data from df (this only contains expected features)
        row_data = df.iloc[i].to_dict()

        # get product id: prefer from df, fallback to original payload dict
        product_id = row_data.get("Item_Identifier")
        if (not product_id) and original_dicts:
            try:
                orig = original_dicts[i]
                product_id = orig.get("Item_Identifier") or orig.get("product_id") or orig.get("ItemId")
            except Exception:
                product_id = None

        # attach predictions
        row_data["predicted_sales"] = pred_vals[i]
        row_data["dynamic_price"] = dynamic_prices[i]
        response_rows.append(row_data)

        # log to DB
        try:
            log_entry = PredictionLog(
                product_id=product_id,
                predicted_sales=pred_vals[i],
                dynamic_price=dynamic_prices[i],
                actual_sales=None
            )
            db.session.add(log_entry)
            app.logger.info("Logging -> product_id=%s, predicted=%s, dynamic=%s",
                            product_id, pred_vals[i], dynamic_prices[i])
        except Exception as ex_log:
            app.logger.error("Failed to create DB log entry: %s", ex_log)

    # commit after adding all entries
    try:
        db.session.commit()
    except Exception as ex_commit:
        app.logger.error("DB commit failed: %s", ex_commit)
        db.session.rollback()

    # return single object if len==1
    if len(response_rows) == 1:
        return response_rows[0]
    return response_rows

# -------------------------
# Endpoints
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": model is not None,
        "model_loaded": model is not None,
        "load_errors": load_errors,
        "model_type": str(type(model)) if model else None
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded - check server logs."}), 500
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400

    payload = request.get_json()

    # get expected features from model if available
    expected_features = None
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        if "Item_Outlet_Sales" in expected_features:
            expected_features = [c for c in expected_features if c != "Item_Outlet_Sales"]

    try:
        # SINGLE sample
        if "features" in payload:
            feat = payload["features"]
            if not isinstance(feat, dict):
                return jsonify({"error": "'features' must be an object/dict"}), 400

            # prepare df and predict
            df = _prepare_dataframe_from_dicts([feat], expected_features)
            preds = model.predict(df)
            response = _make_response_and_log(df, preds, original_dicts=[feat])
            return jsonify(response), 200

        # BATCH samples
        elif "features_list" in payload:
            feats = payload["features_list"]
            if not isinstance(feats, list) or not feats:
                return jsonify({"error": "'features_list' must be a non-empty list"}), 400
            if not all(isinstance(x, dict) for x in feats):
                return jsonify({"error": "Each element in 'features_list' must be a dict"}), 400

            df = _prepare_dataframe_from_dicts(feats, expected_features)
            preds = model.predict(df)
            response = _make_response_and_log(df, preds, original_dicts=feats)
            return jsonify(response), 200

        else:
            return jsonify({"error": "Missing 'features' or 'features_list' in request body"}), 400

    except ValueError as ve:
        app.logger.error("ValueError in predict: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Unexpected exception in predict: %s\n%s", e, tb)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Update actual sales for a prediction log.
    JSON: { "id": <log_id>, "actual_sales": <float> }
    """
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400
    data = request.get_json()
    log_id = data.get("id")
    actual_sales = data.get("actual_sales")

    if not log_id or actual_sales is None:
        return jsonify({"error": "Missing 'id' or 'actual_sales'"}), 400

    log_entry = PredictionLog.query.get(log_id)
    if not log_entry:
        return jsonify({"error": f"No prediction log with id {log_id}"}), 404

    try:
        log_entry.actual_sales = float(actual_sales)
        db.session.commit()
        return jsonify({"status": "updated", "id": log_id, "actual_sales": log_entry.actual_sales}), 200
    except Exception as e:
        db.session.rollback()
        app.logger.error("Failed to update feedback: %s", e)
        return jsonify({"error": "Failed to update feedback", "details": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Compute MAE, RMSE, MAPE for logs where actual_sales is NOT NULL.
    Optional query param: days=<n> to limit to last n days.
    """
    try:
        days = request.args.get("days", None)
        query = PredictionLog.query.filter(PredictionLog.actual_sales.isnot(None))
        if days:
            try:
                days_int = int(days)
                query = query.filter(PredictionLog.created_at >= db.func.datetime('now', f'-{days_int} days'))
            except Exception:
                pass

        logs = query.all()
        if not logs:
            return jsonify({"error": "No logs with actual_sales available"}), 404

        y_true = np.array([l.actual_sales for l in logs], dtype=float)
        y_pred = np.array([l.predicted_sales for l in logs], dtype=float)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        # avoid divide by zero
        nonzero_mask = y_true != 0
        if nonzero_mask.any():
            mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
        else:
            mape = None

        return jsonify({
            "total_samples": int(len(y_true)),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2) if mape is not None else None
        }), 200

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Metrics error: %s\n%s", e, tb)
        return jsonify({"error": "Failed to compute metrics", "details": str(e)}), 500


@app.route("/logs", methods=["GET"])
def logs():
    """
    Return recent logs. Query params:
      - limit (default 50)
      - only_missing_feedback=true to fetch only rows where actual_sales IS NULL
    """
    try:
        limit = int(request.args.get("limit", 50))
        only_missing = request.args.get("only_missing_feedback", "false").lower() in ("1", "true", "yes")
        q = PredictionLog.query
        if only_missing:
            q = q.filter(PredictionLog.actual_sales.is_(None))
        q = q.order_by(PredictionLog.created_at.desc()).limit(limit)
        rows = q.all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "product_id": r.product_id,
                "predicted_sales": r.predicted_sales,
                "dynamic_price": r.dynamic_price,
                "actual_sales": r.actual_sales,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })
        return jsonify(out), 200
    except Exception as e:
        app.logger.error("Failed to fetch logs: %s", e)
        return jsonify({"error": "Failed to fetch logs", "details": str(e)}), 500


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.logger.info("Starting server. MODEL_PATH=%s DB=%s ALPHA=%s", MODEL_PATH, app.config["SQLALCHEMY_DATABASE_URI"], ALPHA)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
