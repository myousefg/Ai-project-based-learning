# =============================================================
# Wine Quality Classification with Manual K‑Nearest Neighbor
# =============================================================

import csv
import math
import random

# 1. Label Handling
def relabel(label: str) -> str:
    return label.strip().lower()

# 2. Data Loading
def load_data(filename: str):
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
    ranges = {}
    keys = ["fixed_acidity", "residual_sugar", "alcohol", "density"]
    for k in keys:
        vals = [row[k] for row in data]
        lo, hi = min(vals), max(vals)
        ranges[k] = (lo, hi)
        for row in data:
            row[k] = (row[k] - lo) / (hi - lo) if hi != lo else 0.0
    return data, ranges

# 4. Train‑Test Split (70‑30)
def split_data(data, ratio: float = 0.7):
    random.seed(42)
    random.shuffle(data)
    cut = int(len(data) * ratio)
    return data[:cut], data[cut:]

# 5. K‑Nearest Neighbor Primitives
def euclidean(a, b):
    return math.sqrt(
        (a["fixed_acidity"]  - b["fixed_acidity"])  ** 2 +
        (a["residual_sugar"] - b["residual_sugar"]) ** 2 +
        (a["alcohol"]        - b["alcohol"])        ** 2 +
        (a["density"]        - b["density"])        ** 2
    )

def majority_vote(labels):
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=counts.get)

def predict_knn(train, sample, k: int = 3):
    distances = [(euclidean(sample, row), row["label"]) for row in train]
    distances.sort(key=lambda tup: tup[0])
    top_k_labels = [lbl for _, lbl in distances[:k]]
    return majority_vote(top_k_labels)

# 6. Evaluation Metrics (macro‑averaged)
def evaluate(y_true, y_pred):
    labels = set(y_true)
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

    print("\n===== Inputkan Spesifikasi Wine =====")
    user = {
        "fixed_acidity":  float(input("Fixed Acidity (4-16): ")),       # g/L
        "residual_sugar": float(input("Residual Sugar (0.5-15): ")),    # g/L
        "alcohol":       float(input("Alcohol (%) (8-14): ")),          # % vol.
        "density":       float(input("Density (0.9900-1.1): ")),        # g/cm³
    }

    for feat in user:
        lo, hi = ranges[feat]
        user[feat] = (user[feat] - lo) / (hi - lo) if hi != lo else 0.0
    prediction = predict_knn(train, user, k)
    print("\nPrediksi kualitas wine:", prediction.upper())

if __name__ == "__main__":
    main()