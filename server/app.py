import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

model = tf.keras.models.load_model("./model/dermavision_final.h5")

class_names = [
    "Acne",
    "Eczema",
    "Normal",
    "Rosacea",
    "Tinea",
    "Vitiligo",
    "Warts"
]

if model.output_shape[-1] != len(class_names):
    print("WARNING: Model output != class_names")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Dermavision API Running", 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((300, 300))

        img_array = np.array(img, dtype=np.float32)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        img_array = np.expand_dims(img_array, axis=0)

    except Exception as e:
        return jsonify({"error": "Image processing failed", "detail": str(e)}), 500

    preds = model.predict(img_array, verbose=0)[0]

    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    if class_idx >= len(class_names):
        return jsonify({
            "error": "Class index out of range",
            "class_idx": class_idx
        }), 500

    label = class_names[class_idx]

    top3_idx = np.argsort(preds)[-3:][::-1]

    top3 = []
    for i in top3_idx:
        if i < len(class_names):
            top3.append({
                "label": class_names[i],
                "confidence": round(float(preds[i]) * 100, 2)
            })

    print("\n===== MODEL OUTPUT =====")
    for i, c in enumerate(class_names):
        print(f"{c}: {preds[i]:.4f}")
    print("========================\n")

    if confidence < 60:
        label = "Unknown / Low Confidence"

    return jsonify({
        "label": label,
        "confidence": round(confidence, 2),
        "top3": top3
    })


if __name__ == "__main__":
    app.run(debug=True)