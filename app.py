from flask import Flask, request, jsonify
from mental_state_analyzer import analyze_image_bytes
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Mental State Analyzer API is running"}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Process the image using your analyzer
    label, conf, reasons, features = analyze_image_bytes(img_bytes)

    return jsonify({
        "label": label,
        "confidence": conf,
        "reasons": reasons,
        "features": features
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
