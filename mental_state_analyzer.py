

# ---------- Imports ----------
import os
import requests
from PIL import Image, ImageOps, ImageFilter
import io
import numpy as np
import time

# ---------- Config ----------
API_KEY=os.environ.get("API_KEY")
API_SECRET=os.environ.get("API_SECRET")
# API_KEY = "v79ft2g4sooqak49j253794905"
# API_SECRET = "36fvadd4h0htf6qrhsqtc9ss33"
SKYBIOMETRY_URL = "https://api.skybiometry.com/fc/faces/detect.json"

# Minimum confidences for using an attribute reliably (tunable)
MIN_ATTR_CONF = {
    "smiling": 40,
    "eyes": 30,
    "lips": 30,
    "age_est": 30,
    "gend$ git add .
git commit -m "Removed invalid syntax from analyzer"
git push -u origin mainer": 30,
    "liveness": 30
}

# Scoring weights (tunable)
WEIGHTS = {
    "downward_pitch": 1.5,
    "tilt_roll": 1.0,
    "no_smile": 1.5,
    "lips_sealed": 1.0,
    "side_yaw": 1.2,
    "eyes_open": 0.8
}

# Confidence threshold for final decision (0..1). Below this -> "uncertain"
FINAL_CONF_THRESHOLD = 0.55

# ---------- Helper: Preprocess Image ----------
def preprocess_image_bytes(img_bytes, max_size=800):
    """
    Improves contrast and resizes the image to help detection.
    Returns bytes of processed JPEG.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # auto-contrast
    img = ImageOps.autocontrast(img)
    # gentle sharpening (helps some faces)
    img = img.filter(ImageFilter.SHARPEN)
    # resize if too large
    w,h = img.size
    if max(w,h) > max_size:
        scale = max_size / float(max(w,h))
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

# ---------- Helper: call SkyBiometry with retries ----------
def call_skybiometry(image_bytes, retries=2, detectors=["aggressive","normal"]):
    """
    POSTs the image to SkyBiometry and returns parsed JSON.
    Will try multiple detector modes and a small pause between retries.
    """
    for detector in detectors[:retries+1]:
        files_payload = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        data = {
            "api_key": API_KEY,
            "api_secret": API_SECRET,
            "attributes": "all",
            "detector": detector
        }
        try:
            resp = requests.post(SKYBIOMETRY_URL, files=files_payload, data=data, timeout=20)
            if resp.status_code == 200:
                j = resp.json()
                if j.get("status") == "success":
                    # If tags is non-empty, return immediately
                    photos = j.get("photos", [])
                    if photos and photos[0].get("tags"):
                        return j
                    # else continue (try next detector or retry)
                else:
                    # API returned failure; still return the JSON for debug
                    return j
            else:
                # Non-200 response; wait and retry
                time.sleep(1.0)
        except Exception as e:
            # network or other error - pause then retry
            time.sleep(1.0)
    return None

# ---------- Helper: Select primary face when multiple faces present ----------
def select_primary_tag(tags):
    """
    Select the tag with the largest area (width * height) assuming values are percentage.
    """
    best = None
    best_area = -1
    for t in tags:
        w = t.get("width") or 0
        h = t.get("height") or 0
        area = (w * h)
        if area > best_area:
            best_area = area
            best = t
    return best

# ---------- Feature extraction with confidence checks ----------
def extract_features_strict(sky_json):
    photos = sky_json.get("photos", [])
    if not photos:
        return None, "no_photos"
    tags = photos[0].get("tags", [])
    if not tags:
        return None, "no_face"
    # pick primary face
    tag = select_primary_tag(tags)
    attributes = tag.get("attributes", {})

    # Safe extraction with confidence thresholds
    def safe_attr_bool(name, treat_none=False):
        if name not in attributes:
            return None, 0
        val = attributes[name].get("value")
        conf = attributes[name].get("confidence", 0)
        if conf < MIN_ATTR_CONF.get(name, 0):
            return None, conf
        # normalize 'true'/'false' strings to 1/0 for known boolean attrs
        if isinstance(val, str):
            if val.lower() == "true":
                return 1, conf
            if val.lower() == "false":
                return 0, conf
        # some attrs like 'smiling' sometimes return percent or bool-like
        try:
            # numeric or text numeric
            iv = int(val)
            return iv, conf
        except:
            return (1 if val else 0) if treat_none else (None, conf)

    def safe_attr_value(name):
        if name not in attributes:
            return None, 0
        val = attributes[name].get("value")
        conf = attributes[name].get("confidence", 0)
        if conf < MIN_ATTR_CONF.get(name, 0):
            return None, conf
        return val, conf

    smiling, smiling_conf = safe_attr_bool("smiling")
    eyes, eyes_conf = safe_attr_value("eyes")   # string: 'open'/'closed'
    lips, lips_conf = safe_attr_value("lips")   # 'sealed' etc.
    liveness, liveness_conf = safe_attr_bool("liveness")
    age, age_conf = safe_attr_value("age_est")
    gender, gender_conf = safe_attr_value("gender")

    # numeric pose values (yaw/roll/pitch) may be present even without attribute confidences
    yaw = tag.get("yaw")
    roll = tag.get("roll")
    pitch = tag.get("pitch")

    features = {
        "yaw": yaw,
        "roll": roll,
        "pitch": pitch,
        "smiling": smiling,
        "smiling_conf": smiling_conf,
        "eyes": eyes,
        "eyes_conf": eyes_conf,
        "lips": lips,
        "lips_conf": lips_conf,
        "liveness": liveness,
        "liveness_conf": liveness_conf,
        "age": int(age) if age and str(age).isdigit() else None,
        "age_conf": age_conf,
        "gender": gender,
        "gender_conf": gender_conf,
        "raw_tag": tag,
        "num_faces": len(tags)
    }

    return features, "ok"

# ---------- Improved rule-based scoring (returns label + confidence + reasons) ----------
def analyze_state_improved(features):
    """
    Uses a weighted scoring of depressive vs daydreaming cues.
    Returns: (label, confidence, reasons_list)
    """
    reasons = []
    # If no features or missing critical pose data -> uncertain
    if features is None:
        return "no_face_detected", 0.0, ["No face data available"]

    # Validate critical pose values
    yaw = features.get("yaw")
    roll = features.get("roll")
    pitch = features.get("pitch")

    # If pose missing, mark uncertain
    if yaw is None and roll is None and pitch is None:
        return "uncertain", 0.0, ["Insufficient pose data"]

    score_depressive = 0.0
    score_daydream = 0.0
    total_possible = 0.0

    # Feature: downward pitch (negative == looking down)
    if isinstance(pitch, (int, float)):
        total_possible += WEIGHTS["downward_pitch"]
        if pitch < -5:  # looking downward
            score_depressive += WEIGHTS["downward_pitch"]
            reasons.append(f"Downward pitch ({pitch}) -> depressive cue")
        else:
            # if looking upward or neutral, daydreaming might benefit little
            if pitch > 5:
                score_daydream += WEIGHTS["downward_pitch"] * 0.2
                reasons.append(f"Upward pitch ({pitch}) weakly supports daydreaming")

    # Feature: head tilt (roll)
    if isinstance(roll, (int, float)):
        total_possible += WEIGHTS["tilt_roll"]
        if abs(roll) > 20:
            score_depressive += WEIGHTS["tilt_roll"] * 0.8
            reasons.append(f"Large head tilt (roll={roll}) -> depressive cue")
        elif abs(roll) > 10:
            score_daydream += WEIGHTS["tilt_roll"] * 0.2
            reasons.append(f"Moderate head tilt (roll={roll}) -> ambiguous cue")

    # Feature: yaw (sideways)
    if isinstance(yaw, (int, float)):
        total_possible += WEIGHTS["side_yaw"]
        if abs(yaw) > 15:
            score_daydream += WEIGHTS["side_yaw"]
            reasons.append(f"Side gaze (yaw={yaw}) -> daydreaming cue")

    # Feature: smiling
    sm = features.get("smiling")
    sm_conf = features.get("smiling_conf", 0) or 0
    if sm is not None:
        total_possible += WEIGHTS["no_smile"]
        if sm == 0:
            score_depressive += WEIGHTS["no_smile"]
            reasons.append(f"No smile detected (conf {sm_conf}) -> depressive cue")
        else:
            score_daydream += WEIGHTS["no_smile"] * 0.2
            reasons.append(f"Smile detected (conf {sm_conf}) -> normal/daydreaming cue")

    # Feature: lips sealed
    lips = features.get("lips")
    lips_conf = features.get("lips_conf", 0) or 0
    if lips is not None:
        total_possible += WEIGHTS["lips_sealed"]
        if str(lips).lower().startswith("sealed"):
            score_depressive += WEIGHTS["lips_sealed"]
            reasons.append(f"Lips sealed (conf {lips_conf}) -> depressive cue")
        else:
            score_daydream += WEIGHTS["lips_sealed"] * 0.1

    # Feature: eyes open/closed
    eyes = features.get("eyes")
    eyes_conf = features.get("eyes_conf", 0) or 0
    if eyes is not None:
        total_possible += WEIGHTS["eyes_open"]
        if str(eyes).lower().startswith("open"):
            score_daydream += WEIGHTS["eyes_open"]
            reasons.append(f"Eyes open (conf {eyes_conf}) -> daydreaming cue")
        else:
            score_depressive += WEIGHTS["eyes_open"] * 0.5
            reasons.append(f"Eyes closed/uncertain (conf {eyes_conf}) -> depressive cue")

    # Normalize to confidence 0..1
    if total_possible <= 0:
        return "uncertain", 0.0, ["No reliable features with sufficient confidence"]
    # compute normalized scores
    norm_dep = score_depressive / total_possible
    norm_day = score_daydream / total_possible
    # also a 'normal' baseline score (if neither depressive nor daydreaming dominate)
    norm_normal = max(0.0, 1.0 - (norm_dep + norm_day))

    # Pick label and confidence as max of normalized scores
    label_scores = {
        "depressive": norm_dep,
        "daydreaming": norm_day,
        "normal": norm_normal
    }
    best_label = max(label_scores, key=lambda k: label_scores[k])
    best_score = label_scores[best_label]

    # Apply final thresholding to avoid low-confidence assertions
    if best_score < FINAL_CONF_THRESHOLD:
        return "uncertain", float(best_score), reasons

    # If multiple faces detected, reduce confidence a bit and mention it
    if features.get("num_faces", 1) > 1:
        reasons.append("Multiple faces detected — prediction uses the largest face; confidence reduced")
        best_score = max(0.0, best_score - 0.1)

    return best_label, float(best_score), reasons

# ---------- Pretty print the model-style output ----------
def print_model_style(features, label, confidence, reasons):
    print("===== MODEL PREDICTION =====")
    if label == "no_face_detected":
        print("Result: NO FACE DETECTED — please upload a clearer face image.")
        return
    if label == "uncertain":
        print("Result: UNCERTAIN — model could not reach reliable confidence.")
        print(f"Confidence: {confidence:.2f}")
        print("Reasons:")
        for r in reasons:
            print(" -", r)
        return

    print(f"Predicted Mental State: {label.upper()}")
    print(f"Confidence: {confidence*100:.1f}%")
    print("\nKey features used:")
    # select and print only the most relevant features we used
    for k in ("yaw","roll","pitch","smiling","smiling_conf","eyes","eyes_conf","lips","lips_conf"):
        if k in features:
            print(f" {k}: {features[k]}")
    print("\nExplanation:")
    for r in reasons:
        print(" -", r)

# ---------- Full pipeline: upload -> preprocess -> call API -> extract -> analyze -> print ----------
def run_full_pipeline_from_upload():
    # Upload image
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded.")
        return
    image_name = next(iter(uploaded))
    original_bytes = uploaded[image_name]
    processed_bytes = preprocess_image_bytes(original_bytes)

    # Call SkyBiometry with retries
    sky_json = call_skybiometry(processed_bytes, retries=2, detectors=["aggressive","normal"])
    if not sky_json:
        print("SkyBiometry request failed or returned no useful data.")
        return

    # Extract strict features
    features, status = extract_features_strict(sky_json)
    if status != "ok":
        # no face or other issue
        print_model_style(None, "no_face_detected", 0.0, [])
        return

    # Analyze
    label, conf, reasons = analyze_state_improved(features)
    # Print
    print_model_style(features, label, conf, reasons)
    return {
        "label": label,
        "confidence": conf,
        "reasons": reasons,
        "features": features,
        "skybiometry_json": sky_json
    }

# ---------- Run ----------
print("Upload an image when prompted. Processing will start automatically.")
result_info = run_full_pipeline_from_upload()
