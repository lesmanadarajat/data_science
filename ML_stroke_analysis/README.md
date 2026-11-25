# 🧠 Brain Stroke Prediction

Prediksi risiko stroke menggunakan Machine Learning (XGBoost + LightGBM + Random Forest ensemble).

## 📊 Dataset
`healthcare-dataset-stroke-data.csv` — 5.110 baris, 12 kolom.

**Target**: `stroke` (0/1) — sangat imbalanced (~4.87% positif).

## 🏗️ Struktur
- `data/` — raw, interim, processed
- `notebooks/` — EDA, preprocessing, modeling, evaluation
- `src/` — source code modular
- `models/` — model artifacts (.pkl)
- `app/` — Flask web app
- `reports/` — figures & metrics

## 🚀 Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Salin dataset
cp healthcare-dataset-stroke-data.csv data/raw/

# 3. Preprocessing
python -m src.data.preprocess

# 4. Training (SMOTE + ensemble)
python -m src.models.train

# 5. Evaluasi
python -m src.models.evaluate

# 6. Jalankan web app
python app/app.py
# Buka http://localhost:5000
```

## 📈 Strategi Akurasi

| Teknik | Alasan |
|---|---|
| **SMOTE oversampling** | Atasi class imbalance (1:20) |
| **Ensemble (XGB+LGBM+RF)** | Robust & stabil |
| **Threshold tuning** | Optimasi F1 untuk recall tinggi |
| **Feature engineering** | glucose_bmi_ratio, age_glucose, dll. |
| **Stratified split** | Jaga proporsi kelas |

## 📁 Metrics Output
Lihat `reports/metrics.txt` untuk hasil evaluasi (ROC-AUC, PR-AUC, F1, Recall, dll.)

## ⚠️ Disclaimer
Model ini untuk **edukasi**, bukan diagnosis medis. Konsultasikan dengan dokter.
