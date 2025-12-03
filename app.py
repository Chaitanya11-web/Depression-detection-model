from flask import Flask, request, jsonify
from mental_state_analyzer import analyze_image_bytes
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)   # allow MERN frontend to call API

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "OK", "message": "Mental State Analyzer model is running"})

@app.route("/analyze", methods=["POST"])
def analyze():
    # Check if image exists
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Run analysis
    try:
        label, conf, reasons, features = analyze_image_bytes(img_bytes)
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

    # Convert features if None or not serializable
    if features is None:
        features = {}

    # Ensure JSON safe (convert numpy/int64/etc. to normal types)
    safe_features = {}
    for k, v in features.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            safe_features[k] = v
        else:
            # Convert unsupported types to string
            safe_features[k] = str(v)

    return jsonify({
        "status": "success",
        "label": label,
        "confidence": float(conf),
        "reasons": reasons,
        "features": safe_features
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
