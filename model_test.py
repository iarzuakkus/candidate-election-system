import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from create_dataset import generate_data
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import io
from matplotlib.figure import Figure
from fastapi.responses import StreamingResponse

def model_report(df, metric: str):
    X = df[['experience', 'technical_test_score']]
    y = df['hired']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    all_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    if isinstance(metric, list):  # güvenlik önlemi
        metric = metric[0] if len(metric) > 0 else "all"

    if metric == "all":
        return all_metrics
    elif metric in all_metrics:
        return {metric: all_metrics[metric]}
    else:
        return {"error": f"Metrik bulunamadı: {metric}"}

def plot_decision_boundary(model, scaler, X, y):
    x_min, x_max = X['experience'].min() - 1, X['experience'].max() + 1
    y_min, y_max = X['technical_test_score'].min() - 5, X['technical_test_score'].max() + 5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = ax.scatter(X['experience'], X['technical_test_score'], c=y, cmap="coolwarm", edgecolors='k')
    ax.set_xlabel("Experience (Years)")
    ax.set_ylabel("Technical Test Score")
    ax.set_title("SVM Model Decision Boundary")
    plt.colorbar(scatter, ax=ax, label='Hired (1=Yes, 0=No)')
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")


if __name__ ==  '__main__':
    df =  generate_data()
    model_reports= model_report(df)
    print(model_reports)

    