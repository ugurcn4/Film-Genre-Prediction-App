# Proje Açıklaması
 Bu proje, kullanıcıların verdiği film özetlerine dayanarak film türlerini tahmin eden bir web uygulamasıdır. Flask tabanlı bir backend kullanarak, makine öğrenmesi modeli sayesinde kullanıcı metinleri üzerinde tahmin yapılır. Model, film türlerini sınıflandırmak için metin madenciliği tekniklerini ve TF-IDF vektörizasyonunu kullanır. Kullanıcılar, film özetlerini girerek bu özetlerin hangi türde olduğunu öğrenebilirler.

 # Proje İçeriği
 Flask tabanlı web uygulaması.
Film türü tahmini için Logistic Regression tabanlı çoklu etiketli sınıflandırma modeli.
Kullanıcıdan metin girişi alarak, film türünü tahmin etme.
Model ve TF-IDF vektörizerinin işlevselliği.
Web arayüzü için HTML ve CSS tasarımı.

# Özellikler
Kullanıcı dostu bir web arayüzü.
Film özetlerine dayalı tür tahminleri.
Yüksek doğruluk oranına sahip bir model ile gerçek zamanlı tahminler.
Kolay kurulabilir ve özelleştirilebilir yapısı.

# Gereksinimler
Python 3.x
Flask
scikit-learn
pandas
joblib
TMDb API

# Kurulum
1. Projeyi bilgisayarınıza klonlayın:
git clone https://github.com/ugurcn4/Film-Genre-Prediction-App

2. Gerekli Python kütüphanelerini yükleyin:
pip install -r requirements.txt
(requirements.txt henüz hazır değil)

3. Flask uygulamasını çalıştırın:
python app.py

4. Tarayıcınızda http://127.0.0.1:5000/ adresine giderek uygulamayı kullanabilirsiniz.


# Katkıda Bulunma
Bu projeye katkı sağlamak isterseniz, lütfen önce bir issue açarak önerilerinizi belirtin.
Pull request göndererek katkıda bulunabilirsiniz.

# Lisans
Bu proje, MIT Lisansı altında lisanslanmıştır.

# Bağlantılar
Repo:
https://github.com/ugurcn4/Film-Genre-Prediction-App
