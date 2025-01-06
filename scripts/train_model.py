from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import joblib
import pandas as pd

# Veriyi yükleyelim
X_train = pd.read_csv("../processed_data/X_train.csv")
X_test = pd.read_csv("../processed_data/X_test.csv")
y_train = pd.read_csv("../processed_data/y_train.csv")
y_test = pd.read_csv("../processed_data/y_test.csv")

# TF-IDF vektörizasyonu
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train["Processed_Özet"])
X_test_tfidf = tfidf.transform(X_test["Processed_Özet"])

# Modeli eğitelim (OneVsRestClassifier ile Logistic Regression)
logreg = LogisticRegression(max_iter=1000)
model = MultiOutputClassifier(logreg, n_jobs=-1)  # Çoklu etiketli sınıflandırma
model.fit(X_train_tfidf, y_train)

# Modeli kaydedelim
joblib.dump(model, '../models/model.pkl')

# TF-IDF vektörizerini kaydedelim
joblib.dump(tfidf, '../models/tfidf_vectorizer.pkl')

# Modeli ve vektörizeri kaydettikten sonra, test seti üzerinde tahmin yapabiliriz
y_pred = model.predict(X_test_tfidf)

# Modelin performansını değerlendirelim
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=y_test.columns))
