import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import os


def clean_iris_data(input_dir='data/raw', output_path='data/processed/iris_processed.csv'):

    iris = load_iris()

    # DataFrame oluşturma (Veri Yükleme Adımı)
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    df_clean = df.copy()

    # Eksik değer kontrolü
    initial_missing = df.isnull().sum().sum()
    if initial_missing > 0:
        print(f"{initial_missing} eksik değer bulundu.")


    # Özellik Mühendisliği ve Kategorik Kodlama

    # 1. Hedef değişkeni (species) Label Encoding ile kodlama
    le_species = LabelEncoder()
    df_clean['species_encoded'] = le_species.fit_transform(df_clean['species'])

    # 2. Yeni özellik oluşturma
    # Sepal (çanak yaprak) alanı hesaplama
    df_clean['sepal_area'] = df_clean['sepal length (cm)'] * df_clean['sepal width (cm)']

    # Petal (taç yaprak) alanı hesaplama
    df_clean['petal_area'] = df_clean['petal length (cm)'] * df_clean['petal width (cm)']

    # Çıktı İşlemleri
    # Çıktı dizinini oluştur (data/processed)
    # os.path.dirname(output_path) -> 'data/processed' kısmını alır.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Temizlenmiş veriyi kaydettik
    df_clean.to_csv(output_path, index=False)


    print(" Iris veri seti temizlendi:", output_path)
    print(f"Orijinal boyut: {df.shape}")
    print(f"Temizlenmiş boyut: {df_clean.shape}")
    print(f"Eksik değerler: {df_clean.isnull().sum().sum()} (eksik değeri olmyan bir veri setidir )")

    # Modelde kullanılacak özellik listesini sıraladım
    features = [
        'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
        'petal width (cm)', 'sepal_area', 'petal_area'
    ]

    print(f"Modelin özellikleri Şu şekildedir : {features}")

    return df_clean, features


if __name__ == "__main__":
    clean_iris_data()