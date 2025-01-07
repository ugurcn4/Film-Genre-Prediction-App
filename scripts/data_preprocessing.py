import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os


def preprocess_data(file_path, output_dir='../processed_data', test_size=0.15, random_state=42):
    # Çıktı dizini oluşturuluyor
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Veriyi yükle
    df = pd.read_csv(file_path)
    print(f"Toplam film sayısı: {len(df)}")

    # Özet bilgisi olmayanları çıkar
    df = df.dropna(subset=['Özet'])
    print(f"Özet bilgisi olan film sayısı: {len(df)}")

    # Metin temizleme
    df['Özet'] = df['Özet'].apply(lambda x: x.lower())  # Özet metnini küçük harfe çevir

    # Türleri listeye dönüştür
    df['Türler'] = df['Türler'].apply(lambda x: [genre.strip() for genre in x.split(',')])

    # Türler için MultiLabelBinarizer kullanarak her türü bir sütun olarak ekleyelim
    genres = ['Aile', 'Aksiyon', 'Animasyon', 'Belgesel', 'Bilim-Kurgu', 'Dram', 'Fantastik',
              'Gerilim', 'Gizem', 'Komedi', 'Korku', 'Macera', 'Müzik', 'Romantik', 'Savaş',
              'Suç', 'TV film', 'Tarih', 'Vahşi Batı']

    mlb = MultiLabelBinarizer(classes=genres)
    y = mlb.fit_transform(df['Türler'])
    y = pd.DataFrame(y, columns=mlb.classes_)

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        df['Özet'],
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Eğitim ve test verilerini kaydetme
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    # MultiLabelBinarizer'ı kaydetme
    joblib.dump(mlb, f"{output_dir}/mlb.pkl")

    print(f"Veri setleri {output_dir} dizinine kaydedildi.")
    print(f"Eğitim seti boyutu: {len(X_train)}")
    print(f"Test seti boyutu: {len(X_test)}")


if __name__ == "__main__":
    file_path = "../data/turkce_filmler_esit_tur.csv"  # CSV dosyanızın yolu
    preprocess_data(file_path)
