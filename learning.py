# Program ini mengimplementasikan Decision Tree untuk klasifikasi kualitas wine
import csv
import math
from collections import Counter

# Fungsi untuk memuat dataset dari file CSV
def load_dataset(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = []
        for row in reader:
            features = list(map(float, row[:-1]))  # Ambil 4 fitur numerik
            label = row[-1].strip().lower()        # Label: low/medium/high
            data.append(features + [label])
    return header, data

# Fungsi untuk menghitung entropy dari dataset
def entropy(data):
    labels = [row[-1] for row in data]
    total = len(labels)
    counts = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

# Fungsi untuk membagi data berdasarkan fitur dan threshold
def split_data(data, feature_index, threshold):
    left = [row for row in data if row[feature_index] <= threshold]
    right = [row for row in data if row[feature_index] > threshold]
    return left, right

# Fungsi untuk mencari split terbaik berdasarkan gain informasi
def best_split(data):
    best_gain = 0
    best_feature = None
    best_threshold = None
    base_entropy = entropy(data)
    n_features = len(data[0]) - 1

    for i in range(n_features):
        unique_vals = sorted(set(row[i] for row in data))
        if len(unique_vals) < 2:
            continue
        thresholds = [(unique_vals[j] + unique_vals[j + 1]) / 2 for j in range(len(unique_vals) - 1)]
        for val in thresholds:
            left, right = split_data(data, i, val)
            if not left or not right:
                continue
            p = len(left) / len(data)
            gain = base_entropy - (p * entropy(left) + (1 - p) * entropy(right))
            if gain > best_gain:
                best_gain = gain
                best_feature = i
                best_threshold = val
    return best_feature, best_threshold

# Kelas Node untuk merepresentasikan setiap node dalam pohon decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

# Fungsi rekursif untuk membangun pohon decision tree
def build_tree(data, depth=0, max_depth=5):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels) or depth == max_depth:
        majority_label = Counter(labels).most_common(1)[0][0]
        return Node(label=majority_label)

    feature, threshold = best_split(data)
    if feature is None:
        majority_label = Counter(labels).most_common(1)[0][0]
        return Node(label=majority_label)

    left_data, right_data = split_data(data, feature, threshold)
    left_branch = build_tree(left_data, depth + 1, max_depth)
    right_branch = build_tree(right_data, depth + 1, max_depth)
    return Node(feature, threshold, left_branch, right_branch)

# Fungsi untuk melakukan prediksi menggunakan pohon decision tree
def predict(tree, row):
    if tree.label is not None:
        return tree.label
    branch = tree.left if row[tree.feature] <= tree.threshold else tree.right
    return predict(branch, row)

# MAIN PROGRAM
if __name__ == "__main__":
    # Load dataset dan bangun pohon
    header, data = load_dataset("wine_quality_classification.csv")
    tree = build_tree(data, max_depth=5)

    # Terima input manual dari user
    print("Masukkan nilai fitur wine:")
    fixed_acidity = float(input("Fixed Acidity (4-16): "))
    residual_sugar = float(input("Residual Sugar (0.5-15): "))
    alcohol = float(input("Alcohol (%) (8-14): "))
    density = float(input("Density (0.9900-1.1): "))

    # Prediksi kualitas wine berdasarkan input user
    user_input = [fixed_acidity, residual_sugar, alcohol, density]
    result = predict(tree, user_input)

    # Tampilkan hasil prediksi
    print(f"\nPrediksi kualitas wine: {result.upper()}")