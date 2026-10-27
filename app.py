from flask import Flask, render_template, request
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.predict import predict

app = Flask(__name__, static_folder="static")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        data = {
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": 1 if request.form["hypertension"] == "1" else 0,
            "heart_disease": 1 if request.form["heart_disease"] == "1" else 0,
            "ever_married": request.form["ever_married"],
            "work_type": request.form["work_type"],
            "Residence_type": request.form["Residence_type"],
            "avg_glucose_level": float(request.form["avg_glucose_level"]),
            "bmi": float(request.form["bmi"]),
            "smoking_status": request.form["smoking_status"],
        }

        # Prediksi dasar
        result = predict(data, threshold=0.10)

        # ⬇️ RULE OVERRIDE: skor risiko klinis
        risk_score = 0
        risk_score += 3 if data["age"] >= 65 else (2 if data["age"] >= 55 else 0)
        risk_score += 2 if data["hypertension"] == 1 else 0
        risk_score += 2 if data["heart_disease"] == 1 else 0
        risk_score += 2 if data["avg_glucose_level"] >= 180 else 0
        risk_score += 1 if data["bmi"] >= 30 else 0
        risk_score += 1 if data["smoking_status"] == "smokes" else 0

        # Boost probabilitas berdasarkan skor risiko
        if risk_score >= 8:
            result["stroke_probability"] = max(result["stroke_probability"], 0.80)
        elif risk_score >= 6:
            result["stroke_probability"] = max(result["stroke_probability"], 0.60)
        elif risk_score >= 4:
            result["stroke_probability"] = max(result["stroke_probability"], 0.35)

        # Update prediction berdasarkan probabilitas baru
        result["prediction"] = int(result["stroke_probability"] >= 0.10)
        result["prob_pct"] = round(result["stroke_probability"] * 100, 2)

        print(f"[debug] Risk score={risk_score}, prob={result['prob_pct']}%")

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)