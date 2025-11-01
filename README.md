# 🌸 Iris Çiçek Türü Sınıflandırma Projesi

Bu repo, Iris veri setini kullanarak çiçek türlerini (Setosa, Versicolor, Virginica) tahmin eden bir makine öğrenimi modeli oluşturmak için yapılmıştır.

Bu proje, veri biliminde standardı olan bazı temel adımları (veri indirme, temizleme, modelleme ve izleme) uygulamayı hedeflemekteyim

## 📁 Proje Yapısından bahsedecek olursak:

Projemizin, tertemiz ve düzenli kalması için standart bir dizin yapısını kullandık:

Bu README.md dosyası, az önce oluşturduğumuz Iris sınıflandırma projemizi özetliyor.

README.md Dosya İçeriği


# 🌸 Iris Çiçek Türü Sınıflandırma Projesi

Merhaba! Bu repo, meşhur Iris veri setini kullanarak çiçek türlerini (Setosa, Versicolor, Virginica) tahmin eden bir makine öğrenimi modeli oluşturmak için yapılmıştır.

Bu proje, veri biliminde standardı olan bazı temel adımları (veri indirme, temizleme, modelleme ve izleme) uygulamayı hedefliyor.

## 📁 Proje Yapısı

Projemiz, temiz ve düzenli kalması için standart bir dizin yapısını kullanır:

sistemimiz bu şekilde ilerliyor önce yükledik temizledik ve train aşamasına geldik 
├── data/
│   ├── raw/                  # Ham (orijinal) veriler (iris.csv)
│   └── processed/            # Temizlenmiş ve işlenmiş veriler
├── models/                   # Eğitilen modeller (.pkl) ve metrikler (.json)
├── src/                      # Tüm Python kodlarımız burada
│   ├── download_data.py      # Veriyi indirdi ve kaydettik
│   ├── clean_data.py         # Veriyi temizledik ve özellik mühendisliği yaptık
│   └── train_model.py        # Modelleri eğittik ve kaydettik
├── requirements.txt          # Gerekli Python kütüphaneleri listesini oluşturduk
└── dvc.yaml                  # DVC ile veri ve aşama takibi
##  Başlangıç (Kurulum aşaması )

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

### 1. Ortamı Hazırlama

Öncelikle gerekli tüm Python kütüphanelerini kuralım.

```bash
 1. Sanal ortamımızı oluşturuyoruz
 venv\Scripts\activate      (Windows)

 2. Gerekli kütüphaneleri requirements.txt dosyasından yükle 
#pycharm eğer önce src kısmı yapılırsa burayı kendi hallediyor.
pip install -r requirements.txt

3. DVC (Data Version Control) Kurulumu
pip install dvc
dvc init

4.Projeyi Çalıştırma
Tüm veri hazırlama, temizleme ve modelleme aşamalarını sırayla çalıştırmak için DVC'nin repro komutunu kullanabiliriz
# dvc.yaml dosyasındaki tüm aşamaları sırasıyla çalıştırır
dvc repro  

