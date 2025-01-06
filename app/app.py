from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Flask uygulamasını başlatıyoruz
app = Flask(__name__)

# Model ve TF-IDF vektörizerini yükleyelim
model = joblib.load('../models/model.pkl')
tfidf = joblib.load('../models/tfidf_vectorizer.pkl')

# Türler listesi (modelde kullanılan türler)
genres = ['Aile', 'Aksiyon', 'Animasyon', 'Belgesel', 'Bilim-Kurgu', 'Dram', 'Fantastik',
          'Gerilim', 'Gizem', 'Komedi', 'Korku', 'Macera', 'Müzik', 'Romantik', 'Savaş',
          'Suç', 'TV film', 'Tarih', 'Vahşi Batı']

# Ana sayfayı render et
@app.route('/')
def home():
    return render_template('index.html')  # Bu, kullanıcıdan metin alacağımız formu barındıracak.

# Tahmin yapacak olan route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Kullanıcıdan gelen metni alalım
        user_input = request.form['text']

        # Metni işleyelim
        user_input_tfidf = tfidf.transform([user_input])

        # Tahmin yapalım
        prediction = model.predict(user_input_tfidf)

        # Tahmin edilen türleri alalım
        predicted_labels = [genre for i, genre in enumerate(genres) if prediction[0][i] == 1]

        return render_template('index.html', prediction=predicted_labels, user_input=user_input)

# Uygulamayı çalıştırıyoruz
if __name__ == "__main__":
    app.run(debug=True)
