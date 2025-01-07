import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib


def train_model(X_train, y_train, output_dir='../models'):
    # TfidfVectorizer kullanarak metinleri sayısal verilere dönüştürme
    tfidf = TfidfVectorizer(stop_words='english')

    # Modeli oluşturma ve eğitme
    models = {}
    for column in y_train.columns:
        print(f"Model eğitiliyor: {column}")

        # Logistic Regression modeli ile eğitim
        model = make_pipeline(tfidf, LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train[column])

        # Modeli kaydetme
        joblib.dump(model, f"{output_dir}/{column}_model.pkl")
        models[column] = model

    # Model ve TFIDF vectorizer'ı kaydetme
    joblib.dump(tfidf, f"{output_dir}/tfidf.pkl")

    print("Modeller başarıyla eğitildi ve kaydedildi.")
    return models, tfidf


def load_data(input_dir='../processed_data'):
    # Eğitim verilerini yükle
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv")

    return X_train['Özet'], y_train


if __name__ == "__main__":
    # Veriyi yükle
    X_train, y_train = load_data()

    # Modeli eğit
    models, tfidf = train_model(X_train, y_train)
