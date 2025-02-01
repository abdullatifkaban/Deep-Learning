# Derin Öğrenme Temelleri

Bu depo, derin öğrenme konusunda sıfırdan ileri seviyeye kadar olan kavramları Türkçe kaynak olarak sunmayı amaçlamaktadır.

## Nasıl Kullanılır?

Bu içerik, konuları zorluk seviyelerine göre aşamalı olarak sunmaktadır:
- 🟢 Başlangıç: Temel kavramlar ve giriş seviyesi uygulamalar
- 🟡 Orta: Orta düzey kavramlar ve pratik uygulamalar
- 🔴 İleri: İleri düzey konular ve kompleks uygulamalar

## İçerik Yapısı

1. Temel Kavramlar 🟢
   > Ön Koşul: Temel Python programlama bilgisi
   - [Yapay Zeka, Makine Öğrenmesi ve Derin Öğrenme](01-Temel-Kavramlar/01-AI-ML-DL.md)
   - [Python ve Gerekli Kütüphaneler](01-Temel-Kavramlar/02-Python-Kutuphaneler.md)
   - [Matematik Temelleri](01-Temel-Kavramlar/03-Matematik-Temelleri.md)

2. Yapay Sinir Ağları 🟢
   > Ön Koşul: Temel matematik (türev, matris işlemleri) ve Python bilgisi
   - [Perceptron ve Sinir Hücreleri](02-Yapay-Sinir-Aglari/01-Perceptron.md)
   - [Çok Katmanlı Ağlar](02-Yapay-Sinir-Aglari/02-Cok-Katmanli-Aglar.md)
   - [Aktivasyon Fonksiyonları](02-Yapay-Sinir-Aglari/03-Aktivasyon-Fonksiyonlari.md)
   - [Geri Yayılım Algoritması](02-Yapay-Sinir-Aglari/04-Geri-Yayilim.md)

3. Derin Öğrenme Modelleri 🟡
   > Ön Koşul: Yapay sinir ağları temelleri, TensorFlow/PyTorch kullanımı
   - [Evrişimli Sinir Ağları (CNN)](03-Derin-Ogrenme-Modelleri/01-CNN.md)
   - [Tekrarlayan Sinir Ağları (RNN)](03-Derin-Ogrenme-Modelleri/02-RNN.md)
   - [LSTM ve GRU](03-Derin-Ogrenme-Modelleri/03-LSTM-GRU.md)
   - [Transformers](03-Derin-Ogrenme-Modelleri/04-Transformers.md)

4. Pratik Uygulamalar 🟡
   > Ön Koşul: CNN, RNN temel bilgisi, veri ön işleme deneyimi
   - [Görüntü Sınıflandırma](04-Pratik-Uygulamalar/01-Goruntu-Siniflandirma.md)
   - [Doğal Dil İşleme](04-Pratik-Uygulamalar/02-Dogal-Dil-Isleme.md)
   - [Nesne Tespiti](04-Pratik-Uygulamalar/03-Nesne-Tespiti.md)
   - [Zaman Serisi Analizi](04-Pratik-Uygulamalar/04-Zaman-Serisi.md)

5. İleri Seviye Konular 🔴
   > Ön Koşul: Derin öğrenme modelleri ve uygulamaları konusunda deneyim
   - [Transfer Öğrenme](05-Ileri-Seviye/01-Transfer-Learning.md)
   - [GANs](05-Ileri-Seviye/02-GANs.md)
   - [Reinforcement Learning](05-Ileri-Seviye/03-Reinforcement-Learning.md)
   - [Model Optimizasyonu](05-Ileri-Seviye/04-Model-Optimization.md)

6. Deployment ve MLOps 🔴
   > Ön Koşul: Python web framework'leri, Docker, temel DevOps bilgisi
   - [Model Deployment ve Servisleştirme](06-Deployment/01-Model-Deployment.md)
   - [Model Monitoring ve MLOps](06-Deployment/02-Model-Monitoring.md)

## Öğrenme Yolu

1. Her bölüme başlamadan önce, belirtilen ön koşulları sağladığınızdan emin olun
2. Konuları sırasıyla takip edin, zorlandığınız yerde önceki konulara dönün
3. Her bölümdeki alıştırmaları mutlaka yapın
4. Pratik uygulamaları kendi projelerinizle pekiştirin

## Yardımcı Kaynaklar

- [Python Programlama Temelleri](https://docs.python.org/3/tutorial/)
- [Matematik için Khan Academy](https://www.khanacademy.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Katkıda Bulunma

Bu projeye katkıda bulunmak için:
1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin yeni-ozellik`)
5. Pull Request oluşturunsen seç ve devam et lütfen

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## Gereksinimler

```bash
pip install tensorflow>=2.0.0
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install fastapi
pip install mlflow
pip install prometheus_client
pip install dvc
```