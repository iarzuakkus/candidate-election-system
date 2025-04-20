#  Candidate Election System

Bu proje, adayların tecrübe yılı ve teknik test puanına göre işe alınıp alınmayacağını tahmin eden bir makine öğrenmesi tabanlı sistemdir. RESTful API mimarisi ile sunulmuştur.

---

##  Özellikler

- Aday verisi üretme (`faker` + `numpy`)
- SVM ile işe alım tahmin modeli
- Model kararı ve sınırı görselleştirme
- Tahmin sonucunu veri setine otomatik ekleme
- FastAPI ile RESTful API servisi
- Swagger UI (otomatik API dokümantasyonu)
- Gelişmiş metrikler: Accuracy, Precision, Recall, F1, Confusion Matrix vs.

---

##  Kullanılan Teknolojiler

- Python 3.12
- Scikit-learn
- FastAPI
- Pandas, NumPy
- Matplotlib
- Faker
- Uvicorn

---

##  Proje Yapısı

```bash
candidate-election-system/
│
├── create_dataset.py        # Sahte aday verisi üretimi (Faker)
├── model_build.py           # Model eğitimi ve kayıt işlemleri
├── model_test.py            # Model değerlendirme ve karar sınırı çizimi
├── main.py                  # FastAPI uygulaması ve endpoint tanımları
└── README.md                # Proje dokümantasyonu
```

---

##  Kurulum

```bash
git clone https://github.com/iarzuakkus/candidate-election-system.git
cd candidate-election-system
```

---

##  Uygulamayı Başlat

```bash
python -m uvicorn main:app --reload
```

- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

---

##  API Endpointleri

| Yöntem | URL             | Açıklama                             |
|--------|------------------|--------------------------------------|
| GET    | `/data`          | Veri setini getir                    |
| GET    | `/data/summary`  | Veri seti hakkında özet              |
| POST   | `/train`         | Mevcut verilerle modeli eğitir       |
| POST   | `/predict`       | Yeni tahmin yapar ve veriye ekler    |
| GET    | `/plot`          | Model karar sınırını görselleştirir  |
| GET    | `/report`        | Seçilen metriklere göre skorları döner |

---

##  Aday Tahmini Örneği (Swagger)

```http
POST /predict?name=Elif%20Kaya&experience=3.2&technical_test_score=76.0
```

---

##  Karar Sınırı Görselleştirilmesi

`/plot` endpointi ile modeli ve karar alanlarını görsel olarak inceleyebilirsiniz.

---

##  Geliştiren

**İlayda Arzu Akkuş**  
[GitHub](https://github.com/iarzuakkus) • Süleyman Demirel Üniversitesi - Bilgisayar Mühendisliği

