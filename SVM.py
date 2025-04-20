import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from create_dataset import generate_data
import joblib

def build_model(df):

    X = df[['experience','technical_test_score']]
    y = df['hired'] 
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel="linear")
    model.fit(X_scaled, y)

    joblib.dump(model,  "model.pkl")
    joblib.dump(scaler, "scaler.pkl")  # Tahmin fonksiyonunda kullanmak uzere scaler nesnesi kaydedildi

    return model, scaler

def model_predict(experience, test_score):
    input_df = pd.DataFrame([{
        'experience' : experience,
        'technical_test_score' : test_score,
    }])
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    return prediction
