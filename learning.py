# =============================================================
# Wine Quality Classification with Manual K‑Nearest Neighbor
# =============================================================

import csv
import math
import random

# 1. Label Handling
def relabel(label: str) -> str:
    """Membersihkan dan menstandarkan label kualitas.

    Dataset sudah pakai kata *low / medium / high*. Kita hanya memastikan
    hasil akhir huruf kecil tanpa spasi ekstra sehingga konsisten di model.
    """
    return label.strip().lower()

# 2. Data Loading
def load_data(filename: str):
    """Membaca CSV → list of dict + konversi tipe data.

    Kolom numerik di‑cast ke *float* agar bisa diolah matematis.
    Label dikirim ke `relabel()` untuk distandarkan.
    """
    data = []
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "fixed_acidity":  float(row["fixed_acidity"]),
                "residual_sugar": float(row["residual_sugar"]),
                "alcohol":       float(row["alcohol"]),
                "density":       float(row["density"]),
                "label":         relabel(row["quality_label"]),
            })
    return data

# 3. Normalization (min‑max)
def normalize(data):
    """Skalakan ke rentang 0‑1 per fitur + simpan rentang awal.

    KNN sensitif terhadap skala: fitur besar bisa mendominasi jarak.
    """
    ranges = {}
    keys = ["fixed_acidity", "residual_sugar", "alcohol", "density"]
    for k in keys:
        vals = [row[k] for row in data]
        lo, hi = min(vals), max(vals)
        ranges[k] = (lo, hi)
        for row in data:
            row[k] = (row[k] - lo) / (hi - lo) if hi != lo else 0.0
    return data, ranges  # data sudah terskala + rentang asli

# 4. Train‑Test Split (70‑30)
def split_data(data, ratio: float = 0.7):
    """Mengacak lalu membagi data → (train, test)."""
    random.seed(42)      # reproducible
    random.shuffle(data) # in‑place shuffle
    cut = int(len(data) * ratio)
    return data[:cut], data[cut:]

# 5. K‑Nearest Neighbor Primitives
def euclidean(a, b):
    """Jarak Euclidean antar dua sampel (4 dimensi)."""
    return math.sqrt(
        (a["fixed_acidity"]  - b["fixed_acidity"])  ** 2 +
        (a["residual_sugar"] - b["residual_sugar"]) ** 2 +
        (a["alcohol"]        - b["alcohol"])        ** 2 +
        (a["density"]        - b["density"])        ** 2
    )

def majority_vote(labels):
    """Mengembalikan label terbanyak di *labels* (tanpa Counter)."""
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=counts.get)

def predict_knn(train, sample, k: int = 3):
    """Prediksi satu *sample* menggunakan KNN dengan k=3 (default)."""
    # 1) Hitung jarak ke setiap data training
    distances = [(euclidean(sample, row), row["label"]) for row in train]
    # 2) Urutkan dari jarak terkecil
    distances.sort(key=lambda tup: tup[0])
    # 3) Ambil k label terdekat
    top_k_labels = [lbl for _, lbl in distances[:k]]
    return majority_vote(top_k_labels)

# 6. Evaluation Metrics (macro‑averaged)
def evaluate(y_true, y_pred):
    """Hitung akurasi, presisi, recall, F1 (macro)."""
    labels = set(y_true)
    # ---- Accuracy ----
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    precision_sum = recall_sum = f1_sum = 0
    for lbl in labels:
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred):
            if t == lbl and p == lbl:
                tp += 1
            elif t != lbl and p == lbl:
                fp += 1
            elif t == lbl and p != lbl:
                fn += 1
        prec_lbl = tp / (tp + fp) if (tp + fp) else 0
        recall_lbl = tp / (tp + fn) if (tp + fn) else 0
        f1_lbl = 2 * prec_lbl * recall_lbl / (prec_lbl + recall_lbl) if (prec_lbl + recall_lbl) else 0
        precision_sum += prec_lbl
        recall_sum += recall_lbl
        f1_sum += f1_lbl

    n = len(labels)
    return {
        "accuracy":  accuracy,
        "precision": precision_sum / n,
        "recall":    recall_sum / n,
        "f1_score":  f1_sum / n,
    }

# 7. Main Program
def main():
    data = load_data("wine_quality_classification.csv")
    data, ranges = normalize(data)
    train, test = split_data(data, 0.7)
    k = 3  

    y_true_train = [row["label"] for row in train]
    y_pred_train = [predict_knn(train, row, k) for row in train]

    y_true_test = [row["label"] for row in test]
    y_pred_test = [predict_knn(train, row, k) for row in test]

    print("Evaluasi Training:")
    for metric, val in evaluate(y_true_train, y_pred_train).items():
        print(f"{metric.capitalize():10}: {val:.4f}")

    print("\nEvaluasi Testing:")
    for metric, val in evaluate(y_true_test, y_pred_test).items():
        print(f"{metric.capitalize():10}: {val:.4f}")

    print("\n===== DEMO =====")
    user = {
        "fixed_acidity":  float(input("Fixed acidity: ")),  # g/L
        "residual_sugar": float(input("Residual sugar: ")), # g/L
        "alcohol":       float(input("Alcohol (%): ")),     # % vol.
        "density":       float(input("Density: ")),         # g/cm³
    }

    for feat in user:
        lo, hi = ranges[feat]
        user[feat] = (user[feat] - lo) / (hi - lo) if hi != lo else 0.0

    prediction = predict_knn(train, user, k)
    print("\nPrediksi kualitas wine:", prediction.upper())


if __name__ == "__main__":
    main()