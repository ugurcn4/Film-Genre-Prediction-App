from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

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


@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    input_text = ""

    if request.method == 'POST':
        # Kullanıcıdan gelen metni alın
        input_text = request.form['input_text']

        # Tahmin yap
        predictions = load_model_and_predict(input_text)

    return render_template('index.html', predictions=predictions, input_text=input_text)


if __name__ == "__main__":
    app.run(debug=True)
