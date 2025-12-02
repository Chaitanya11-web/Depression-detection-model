

# ---------- Imports ----------
import requests
from PIL import Image, ImageOps, ImageFilter
import io
import numpy as np
import time
import os

# ---------- Config ----------
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
SKYBIOMETRY_URL = "https://api.skybiometry.com/fc/faces/detect.json"

# Minimum confidences
MIN_ATTR_CONF = {
    "smiling": 40,
    "eyes": 30,
    "lips": 30,
    "age_est": 30,
    "gender": 30,
    "liveness": 30
}

# Scoring weights
WEIGHTS = {
    "downward_pitch": 1.5,
    "tilt_roll": 1.0,
    "no_smile": 1.5,
    "lips_sealed": 1.0,
    "side_yaw": 1.2,
    "eyes_open": 0.8
}

FINAL_CONF_THRESHOLD = 0.55

# ---------- Preprocess image ----------
def preprocess_image_bytes(img_bytes, max_size=800):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

# ---------- SkyBiometry API Call ----------
def call_skybiometry(image_bytes, retries=2, detectors=["aggressive", "normal"]):
    for detector in detectors[:retries+1]:
        payload = {
            "api_key": API_KEY,
            "api_secret": API_SECRET,
            "attributes": "all",
            "detector": detector
        }
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

        try:
            resp = requests.post(SKYBIOMETRY_URL, data=payload, files=files, timeout=20)
            if resp.status_code == 200:
                j = resp.json()
                photos = j.get("photos", [])
                if photos and photos[0].get("tags"):
                    return j
            time.sleep(1)
        except:
            time.sleep(1)
    return None

# ---------- Select major face ----------
def select_primary_tag(tags):
    best = None
    max_area = -1
    for t in tags:
        area = (t.get("width", 0) * t.get("height", 0))
        if area > max_area:
            max_area = area
            best = t
    return best

# ---------- Extract features ----------
def extract_features_strict(sky_json):
    photos = sky_json.get("photos", [])
    tags = photos[0].get("tags", []) if photos else []

    if not tags:
        return None, "no_face"

    tag = select_primary_tag(tags)
    attributes = tag.get("attributes", {})

    def safe_value(name):
        if name not in attributes:
            return None, 0
        val = attributes[name]["value"]
        conf = attributes[name]["confidence"]
        if conf < MIN_ATTR_CONF.get(name, 0):
            return None, conf
        return val, conf

    smiling, sm_conf = safe_value("smiling")
    eyes, eyes_conf = safe_value("eyes")
    lips, lips_conf = safe_value("lips")
    liveness, liveness_conf = safe_value("liveness")

    yaw = tag.get("yaw")
    roll = tag.get("roll")
    pitch = tag.get("pitch")

    features = {
        "yaw": yaw,
        "roll": roll,
        "pitch": pitch,
        "smiling": 0 if smiling == "false" else 1,
        "smiling_conf": sm_conf,
        "eyes": eyes,
        "eyes_conf": eyes_conf,
        "lips": lips,
        "lips_conf": lips_conf,
        "num_faces": len(tags)
    }

    return features, "ok"

# ---------- Analyze state ----------
def analyze_state_improved(features):
    if not features:
        return "no_face_detected", 0.0, ["No detectable features"]

    yaw = features["yaw"]
    roll = features["roll"]
    pitch = features["pitch"]

    sm = features["smiling"]
    sm_conf = features["smiling_conf"]

    reasons = []
    score_dep = 0
    score_day = 0
    total = 0

    if pitch is not None:
        total += WEIGHTS["downward_pitch"]
        if pitch < -5:
            score_dep += WEIGHTS["downward_pitch"]
            reasons.append(f"Downward pitch ({pitch}) → depressive cue")

    if abs(roll) > 20:
        total += WEIGHTS["tilt_roll"]
        score_dep += WEIGHTS["tilt_roll"]
        reasons.append(f"Tilted head (roll {roll}) → depressive cue")

    if abs(yaw) > 15:
        total += WEIGHTS["side_yaw"]
        score_day += WEIGHTS["side_yaw"]
        reasons.append(f"Side gaze ({yaw}) → daydreaming cue")

    if sm == 0:
        total += WEIGHTS["no_smile"]
        score_dep += WEIGHTS["no_smile"]
        reasons.append(f"No smile (conf {sm_conf}) → depressive cue")

    if total == 0:
        return "uncertain", 0.0, reasons

    conf_dep = score_dep / total
    conf_day = score_day / total
    conf_normal = max(0, 1 - (conf_dep + conf_day))

    scores = {
        "depressive": conf_dep,
        "daydreaming": conf_day,
        "normal": conf_normal
    }

    label = max(scores, key=scores.get)
    conf = scores[label]

    if conf < FINAL_CONF_THRESHOLD:
        return "uncertain", conf, reasons

    return label, conf, reasons

# ---------- PUBLIC FUNCTION (USED BY Flask API) ----------
def analyze_image_bytes(image_bytes):
    processed = preprocess_image_bytes(image_bytes)
    sky_json = call_skybiometry(processed)

    if not sky_json:
        return "no_face_detected", 0.0, [], None

    features, status = extract_features_strict(sky_json)

    if status != "ok":
        return "no_face_detected", 0.0, [], None

    label, conf, reasons = analyze_state_improved(features)
    return label, conf, reasons, features
