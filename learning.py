import csv
import math
import random
from collections import Counter, defaultdict

def load_dataset(filename):
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  
            data = []
            for row in reader:
                try:
                    features = list(map(float, row[:-1]))
                    label = row[-1].strip().lower() 
                    data.append(features + [label])
                except ValueError as e:
                    print(f"Peringatan: Melewatkan baris karena kesalahan konversi data: {row} - {e}")
                    continue
        return header, data
    except FileNotFoundError:
        print(f"Error: File '{filename}' tidak ditemukan. Pastikan file berada di direktori yang sama atau berikan path lengkap.")
        return None, None
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat dataset: {e}")
        return None, None

def train_test_split(data, test_size=0.2):
    random.seed(42) 
    random.shuffle(data)
    
    train_split_index = int(len(data) * (1 - test_size)) 
    train_data = data[:train_split_index]
    test_data = data[train_split_index:]
    return train_data, test_data

def euclidean_distance(row1, row2):
    distance = 0.0
    
    for i in range(len(row1) - 1): 
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

def get_neighbors(training_data, test_row, k):
    distances = []
    for train_row in training_data:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda x: x[1]) 
    
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0]) 
    return neighbors

def knn_predict(training_data, test_row, k):
    neighbors = get_neighbors(training_data, test_row, k)
    output_values = [row[-1] for row in neighbors] 
    
    prediction = Counter(output_values).most_common(1)[0][0]
    return prediction

def calculate_metrics(model_training_data, dataset, k_value, dataset_name="Data"):
    if not dataset:
        print(f"Tidak ada data di {dataset_name} untuk dievaluasi.")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

    true_labels = [row[-1] for row in dataset]
    predicted_labels = []
    for row in dataset:
        features = row[:-1] 
        predicted_labels.append(knn_predict(model_training_data, row, k_value))

    all_possible_labels = sorted(list(set(true_labels + predicted_labels)))
    
    tp = defaultdict(int) # True Positives
    fp = defaultdict(int) # False Positives
    fn = defaultdict(int) # False Negatives
    
    for true_l, pred_l in zip(true_labels, predicted_labels):
        if true_l == pred_l:
            tp[true_l] += 1
        else:
            fp[pred_l] += 1 
            fn[true_l] += 1
            
    precisions = []
    recalls = []
    f1_scores = []
    
    total_correct = 0
    for label in all_possible_labels:
        
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        total_correct += tp[label]

    accuracy = total_correct / len(dataset)
    
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print(f"\n--- Evaluasi Model pada {dataset_name} ---")
    print(f"Akurasi: {accuracy:.2%}")
    print(f"Precision (Macro Avg): {avg_precision:.2f}")
    print(f"Recall (Macro Avg): {avg_recall:.2f}")
    print(f"F1-Score (Macro Avg): {avg_f1_score:.2f}")
    
    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1_score
    }

if __name__ == "__main__":
    filename = "wine_quality_classification.csv" # Nama file dataset Anda
    K_VALUE = 5 # Anda bisa mengubah nilai K ini untuk KNN

    print(f"Memuat dataset dari '{filename}'...")
    header, all_data = load_dataset(filename)
    
    if all_data is None: # Jika gagal memuat data, keluar
        print("Gagal memuat dataset. Program dihentikan.")
    else:
        print(f"Dataset berhasil dimuat dengan {len(all_data)} sampel.")
        
        train_data, test_data = train_test_split(all_data, test_size=0.2) 
        print(f"Data dibagi: {len(train_data)} sampel untuk training (80%), {len(test_data)} sampel untuk testing (20%).")

        print(f"Model K-Nearest Neighbors disiapkan dengan K = {K_VALUE}.")

        calculate_metrics(train_data, train_data, K_VALUE, "Data Latih")

        calculate_metrics(train_data, test_data, K_VALUE, "Data Uji")

        print("\n--- Demo Pengujian Model & Prediksi Kualitas Wine (Input Manual) ---")
        print("Masukkan nilai fitur wine untuk prediksi (gunakan titik untuk desimal):")
        
        feature_prompts = [f"{f.replace('_', ' ').title()}: " for f in header[:-1]]
        
        user_input_features = []

        for i, prompt in enumerate(feature_prompts):
            while True:
                try:
                    value = float(input(prompt))
                    user_input_features.append(value)
                    break
                except ValueError:
                    print("Input tidak valid. Harap masukkan angka.")

        user_input_full_row = user_input_features + ["dummy_label"] 
        result = knn_predict(train_data, user_input_full_row, K_VALUE)

        print(f"\nPrediksi kualitas wine berdasarkan input Anda: {result.upper()}")
