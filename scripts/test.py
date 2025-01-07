import joblib
import pandas as pd


def load_model_and_predict(input_text, model_dir='../models'):
    # Model ve TFIDF vectorizer'ı yükleme
    tfidf = joblib.load(f"{model_dir}/tfidf.pkl")
    models = {}

    genres = ['Aile', 'Aksiyon', 'Animasyon', 'Belgesel', 'Bilim-Kurgu', 'Dram', 'Fantastik',
              'Gerilim', 'Gizem', 'Komedi', 'Korku', 'Macera', 'Müzik', 'Romantik', 'Savaş',
              'Suç', 'TV film', 'Tarih', 'Vahşi Batı']

    # Her bir tür için model yükleme ve tahmin yapma
    for genre in genres:
        model = joblib.load(f"{model_dir}/{genre}_model.pkl")
        prediction = model.predict([input_text])
        models[genre] = prediction[0]

    return models


if __name__ == "__main__":
    # Örnek bir özet
    input_text = """
    Gerçek bir hayat hikayesinden uyarlanan filmde Henry Hill adında bir gangster, Jimmy Conway ve Tommy De Vito adlı iki arkadaşıyla birlikte bir soyguna kalkışır. Gözleri daha yukarda olan iki arkadaşı soyguna katılan diğerlerini öldürür ve mafya içinde yükselmeye başlarlar. Bu durum Henry'i olumsuz etkilemiştir ve bu konuda birşeyler yapması gerekmektedir. Büyük usta Martin Scorsese'nin başyapıtlarından biri olan Goodfellas, 1991 yılında 6 dalda Oscar'a aday gösterilmiş, en iyi yardımcı erkek oyuncu dalında Joe Pesci'ye ödül kazandırmıştı.
    """

    # Modeli yükle ve tahmin yap
    predictions = load_model_and_predict(input_text)

    print("Tahmin Sonuçları:")
    for genre, prediction in predictions.items():
        print(f"{genre}: {'Var' if prediction == 1 else 'Yok'}")
