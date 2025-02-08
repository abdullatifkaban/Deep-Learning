# Yapay Sinir Ağlarının Temel Yapısı

## 1. Yapay Sinir Ağlarının Temelleri

Yapay sinir ağları (YSA), biyolojik sinir sistemlerinden esinlenerek geliştirilmiş hesaplama modelleridir. Temel amacı, veriler arasındaki ilişkileri öğrenmek ve genelleştirme yapabilmektir. Yapay sinir ağları, birçok katmandan oluşan bir yapıya sahiptir.

### 2. Yapay Sinir Ağı Katmanları

Bir yapay sinir ağı genellikle üç temel katmandan oluşur:

- **Giriş Katmanı:** Verilerin modele alındığı katmandır. Her düğüm bir girdiyi temsil eder.
- **Gizli Katman(lar):** Girdileri işleyen ve özellikleri öğrenen katmanlardır.
- **Çıkış Katmanı:** Son tahmin veya sınıflandırmayı gerçekleştiren katmandır.

Aşağıdaki görsel, temel bir yapay sinir ağı modelini göstermektedir:

![Yapay Sinir Ağı Şeması](https://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg)

## 3. Yapay Nöron Modeli

Yapay sinir ağlarında her nöron, girişlerden gelen bilgiyi ağırlıklarla çarpar, toplar ve bir aktivasyon fonksiyonuna uygular. Matematiksel olarak aşağıdaki gibi ifade edilir:

$$ z = \sum_{i=1}^{n} w_i x_i + b $$

Burada:
- $ x_i $: Giriş değerleri
- $ w_i $: Ağırlık katsayıları
- $ b $: Bias terimi
- $ z $: Net giriş değeri

Bu net giriş, bir aktivasyon fonksiyonuna uygulanarak çıkış değeri hesaplanır:

$$ y = f(z) $$

### 4. Aktivasyon Fonksiyonları

Aktivasyon fonksiyonları, bir nöronun çıkışını belirleyen matematiksel fonksiyonlardır. En yaygın kullanılan aktivasyon fonksiyonları şunlardır:

* Sigmoid Fonksiyonu
* ReLU (Rectified Linear Unit) Fonksiyonu
* Tanh Fonksiyonu
* Softmax Fonksiyonu

## 5. İleri Beslemeli Sinir Ağı

İleri beslemeli sinir ağı (Feedforward Neural Network - FNN), verinin giriş katmanından çıkış katmanına doğru aktığı bir yapıdır. Her katmandaki nöronlar, bir önceki katmandan gelen bilgiyi işler ve bir sonraki katmana iletir.

Bu tür ağlar genellikle aşağıdaki gibi gösterilir:

![İleri Beslemeli Ağ](https://mukulrathi.com/static/648e5207805f95bf09c330a43d89d295/f207c/neural-net.png)

## 6. Sonuç

Yapay sinir ağlarının temel yapısını anlamak, derin öğrenme modellerini geliştirmek için kritik bir adımdır. Katmanların düzenlenmesi, ağırlıkların öğrenilmesi ve aktivasyon fonksiyonlarının seçimi, modelin başarısını doğrudan etkiler. 

## 📚 Önerilen Kaynaklar
- [Neuroscience Online - Chapter 1: Neuronal Structure](https://nba.uth.tmc.edu/neuroscience/s1/chapter01.html)
- [From Biological to Artificial Neurons - Nature Reviews](https://www.nature.com/articles/s41583-021-00455-7)
- [Neural Networks and Deep Learning - Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html)
