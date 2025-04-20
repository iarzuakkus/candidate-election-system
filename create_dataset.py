import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

fake = Faker('tr_TR')

#Fake basvuru tarihi icin baslangic ve bitis tarihleri belirlendi
s_date = datetime.strptime('2025-03-01', '%Y-%m-%d').date()
e_date = datetime.strptime('2025-03-31', '%Y-%m-%d').date()

def generate_data(n_samples = 3000, random_state = 42):
    np.random.seed(random_state)
    fake.seed_instance(random_state)

    # Fake 3000 isim-soyisim verisi olusturuldu
    names = [fake.name() for _ in range(n_samples)]

    # Fake 3000 basvuru tarih verisi olusturuldu
    application_dates = [fake.date_between(start_date=s_date, end_date=e_date) for _ in range(n_samples)]

    # Rastgele deneyim yili olusturuldu
    experience_years = np.random.randint(0,10,n_samples)

    # Rastgele teknik test sonucu olusturuldu
    technical_test_score = np.random.uniform(0,100,n_samples)

    labels = [
    1 if exp >= 2 and score >= 60 else 0
    for exp, score in zip(experience_years, technical_test_score)
]

    # Gurultu eklemek icin rastgele indisleri secilir
    noise_indices = np.random.choice(n_samples, size= 100, replace=False)

    # Label icin gurultu ekleme (label = 0 ise 1, label = 1 ise 0 yapacagiz)
    for i in noise_indices:
        labels[i] = 1 - labels[i]

    df = pd.DataFrame(
        {
            'name' : names,
            'application_date' : application_dates,
            'experience' : experience_years,
            'technical_test_score' : technical_test_score,
            'hired' : labels
        }
    )

    return df

if __name__ ==  '__main__':
    df = generate_data()
    print(df)