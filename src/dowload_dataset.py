import pandas as pd
from sklearn.datasets import load_iris  # Iris veri setini yüklemek için sklearn'ı kullanıyoruz
import os


def download_iris_data():
    """Scikit-learn kütüphanesinde hazır veriyi kullandım"""

    # 1. Veri dizinlerini (data/raw) oluşturma
    # exist_ok=True:klasörümüzü oluşturduk eğer varsa da hata vermez.
    os.makedirs('data/raw', exist_ok=True)

    # 2. Scikit-learn'den Iris veri setini yükleme
    iris = load_iris()

    # Veriyi Pandas DataFrame'e dönüştürme
    # data.data: Özellikler (sepal: uzunluk /petal:genişlikleri)
    # data.feature_names: Özellik isimleri
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Hedef değişkeni (Tür) ekleme
    # data.target: Sayısal tür etiketleri (0, 1, 2)
    df['species_code'] = iris.target

    # Tür isimlerini (Setosa, Versicolor, Virginica) anlaşılır hale getirme işlemi
    df['species'] = df['species_code'].apply(lambda x: iris.target_names[x])

    # 3. Ham veriyi CSV olarak kaydetme işlemi
    df.to_csv('data/raw/iris.csv', index=False)

    print("Iris veri seti başarıyka indirildi : data/raw/iris.csv")
    print(f"Veri boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")

    # Iris veri setinde eksik değer yok temiz dataset
    print("\nVerinin İlk 5 Satırı Aşağıdadır :")
    print(df.head())

    return df


if __name__ == "__main__":
    download_iris_data()