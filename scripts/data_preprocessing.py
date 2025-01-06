import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download("punkt_tab")
nltk.download("stopwords")

# Türkçe stopwords
STOPWORDS = set(stopwords.words("turkish"))

# Veri önişleme fonksiyonları
def clean_text(text):
    """
    Metin temizleme fonksiyonu:
    - Küçük harfe çevirir.
    - Noktalama işaretlerini kaldırır.
    - Türkçe stopwords (gereksiz kelimeleri) temizler.
    """
    text = text.lower()  # Küçük harf
    text = re.sub(r"[^\w\s]", "", text)  # Noktalama işaretlerini kaldır
    tokens = word_tokenize(text)  # Tokenize et
    tokens = [word for word in tokens if word not in STOPWORDS]  # Stopwords temizle
    return " ".join(tokens)

def preprocess_data(file_path):
    """
    Veri setini işler:
    - Özetleri temizler.
    - Türleri sayısal forma çevirir.
    - Eksik verileri (özet veya tür) içeren satırları siler.
    """
    # CSV dosyasını oku
    df = pd.read_csv(file_path)

    # Başlangıçtaki satır sayısını kaydet
    initial_row_count = len(df)

    # Tür veya özet eksik olan satırları sil
    df = df.dropna(subset=["Özet", "Türler"])

    # Silinen satır sayısını hesapla
    deleted_rows = initial_row_count - len(df)

    # Özetleri temizle
    df["Processed_Özet"] = df["Özet"].apply(clean_text)

    # Türleri listeye çevir (MultiLabel Binarization için)
    df["Türler_Listesi"] = df["Türler"].apply(lambda x: x.split(", "))

    # Türleri binarize et
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["Türler_Listesi"])

    # X (özetler) ve y (etiketler) olarak ayır
    X = df["Processed_Özet"]
    y = pd.DataFrame(y, columns=mlb.classes_)

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Veriyi CSV dosyasına kaydet
    X_train.to_csv("../processed_data/X_train.csv", index=False)
    X_test.to_csv("../processed_data/X_test.csv", index=False)
    y_train.to_csv("../processed_data/y_train.csv", index=False)
    y_test.to_csv("../processed_data/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test, mlb.classes_, deleted_rows

# Ana fonksiyon
if __name__ == "__main__":
    # Veri seti dosya yolu
    file_path = "../data/turkce_filmler.csv"

    # Veriyi işle
    X_train, X_test, y_train, y_test, classes, deleted_rows = preprocess_data(file_path)

    # Örnek çıktı
    print("Eğitim veri sayısı:", len(X_train))
    print("Test veri sayısı:", len(X_test))
    print("Türler:", classes)
    print("İşlenmiş bir örnek metin:", X_train.iloc[0])

    # Silinen satır sayısı
    print(f"Silinen satır sayısı: {deleted_rows}")
