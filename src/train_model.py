import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os


def train_iris_model(model_type='random_forest'):

    try:
        # İşlenmiş veriyi yükle
        df = pd.read_csv('data/processed/iris_processed.csv')
    except FileNotFoundError:
        print("HATA: 'data/processed/iris_processed.csv' dosyası bulunamadı.")
        print("Lütfen önce veri indirme ve temizleme fonksiyonlarını çalıştırın.")
        return None, None

    # X (Özellikler) ve y (Hedef) değişkenlerini ayır
    feature_cols = [
        'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
        'petal width (cm)', 'sepal_area', 'petal_area'
    ]

    X = df[feature_cols]
    # Hedef değişkeni Iris'te tür (species_encoded)
    y = df['species_encoded']

    # Train-test split (Eğitim ve Test verisini ayırma)
    # stratify=y ile her sette türlerin dağılımı korunur.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Model seç
    if model_type == 'random_forest':
        # Random Forest Sınıflandırıcı
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    else:  # logistic_regression
        # Lojistik Regresyon Sınıflandırıcı
        model = LogisticRegression(
            random_state=42,
            max_iter=2000
        )

    # Model eğit (fit et)
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)

    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)

    # Model kaydet
    os.makedirs('models', exist_ok=True)
    model_path = f'models/iris_{model_type}_model.pkl'
    joblib.dump(model, model_path)

    # Metrikleri kaydet (JSON dosyasına)
    metrics = {
        'model_type': model_type,
        'accuracy': float(accuracy),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    # Metrikler dosyasını kaydet
    with open(f'models/iris_{model_type}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Özellik listesini kaydet
    with open('models/iris_features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Iris modeli eğitildi: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model kaydedildi: {model_path}")

    # Detaylı rapor
    print("\n Classification Report (Kesinlik, Duyarlılık, F1-Score):")
    print(classification_report(y_test, y_pred))

    return model, metrics


if __name__ == "__main__":
    # Random Forest modeli eğitmeyi deniyoruz
    model_rf, metrics_rf = train_iris_model('random_forest')

    print("-" * 50)

    # Logistic Regression modelini deniyoruz
    model_lr, metrics_lr = train_iris_model('logistic_regression')

    print("\n Her iki Iris modeli eğitildi")

    #sonuçlara göre lojistik regresyon daha iyi bir accuracy değeri verdi her iki modelde de Iris Versicolor (1) lerin yakalanma oranı düşük çıktı
    #Model, gerçekte Versicolor olan vakaların sadece %88'ini doğru tahmin etmiş. Kalan %12'lik kısmı FN kalmış