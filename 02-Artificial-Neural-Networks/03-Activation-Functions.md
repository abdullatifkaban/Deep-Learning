# Aktivasyon Fonksiyonları

## 1. Aktivasyon Fonksiyonlarının Önemi

Aktivasyon fonksiyonları, yapay sinir ağlarında her nöronun çıkışını belirleyen matematiksel fonksiyonlardır. Temel olarak, girdileri alır ve belirli bir aralıkta çıktı üretirler. Aktivasyon fonksiyonları, ağın öğrenme kapasitesini ve doğruluğunu doğrudan etkileyen önemli bileşenlerdir.

## 2. Yaygın Aktivasyon Fonksiyonları

### a) Sigmoid Fonksiyonu
Sigmoid fonksiyonu, çıktı değerini 0 ile 1 arasına sıkıştıran bir fonksiyondur:

$$ f(z) = \frac{1}{1+e^{-z}} $$

**Özellikler:**
- Çıktı aralığı: (0,1)
- Olasılık tahmini yapan modellerde yaygın kullanılır.
- Gradyan kaybolma (vanishing gradient) problemine yol açabilir.

**Sigmoid Grafiği:**  
![Sigmoid Grafiği](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

[Kaynak](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

---

### b) ReLU (Rectified Linear Unit) Fonksiyonu
ReLU fonksiyonu, negatif değerleri sıfıra ayarlarken pozitif değerleri olduğu gibi bırakır:

$$ f(z) = \max(0, z) $$

**Özellikler:**
- Çıktı aralığı: $ [0, +\infty) $
- Hesaplama açısından verimli ve gradyan kaybolma sorununu azaltır.
- Negatif girdiler için gradyan sıfır olduğu için "ölü nöron" problemi olabilir.

**ReLU Grafiği:**  
![ReLU Grafiği](https://www.nomidl.com/wp-content/uploads/2022/04/image-10.png)

[Kaynak](https://www.nomidl.com/wp-content/uploads/2022/04/image-10.png)

---

### c) Tanh Fonksiyonu
Tanh fonksiyonu, sigmoid fonksiyonuna benzer ancak çıkış aralığı -1 ile 1 arasındadır:

$$ f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $$

**Özellikler:**
- Çıktı aralığı: $ (-1,1) $
- Sıfır merkezlidir ve genellikle sigmoid’den daha iyi performans gösterir.

**Tanh Grafiği:**  
![Tanh Grafiği](https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/2560px-Hyperbolic_Tangent.svg.png)

[Kaynak](https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/2560px-Hyperbolic_Tangent.svg.png)

---

### d) Softmax Fonksiyonu
Softmax fonksiyonu, çok sınıflı sınıflandırma problemlerinde kullanılır ve her sınıfa ait olasılığı hesaplar:

$$ f(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$

**Özellikler:**
- Sınıflandırma problemlerinde çıktıların toplamını 1 yaparak olasılık dağılımı oluşturur.
- Genellikle son katmanda kullanılır.

**Softmax Grafiği:**

![Softmax Grafiği](https://miro.medium.com/v2/resize:fit:480/1*5nKWsukS6lPR-7fHtlK2Rg.png)

[Kaynak](https://miro.medium.com/v2/resize:fit:480/1*5nKWsukS6lPR-7fHtlK2Rg.png)

**Softmax hakkında daha fazla bilgi:**  
[Softmax Fonksiyonu](https://en.wikipedia.org/wiki/Softmax_function)

## 3. Sonuç
Aktivasyon fonksiyonları, yapay sinir ağlarının başarısını doğrudan etkileyen temel bileşenlerdir. Hangi fonksiyonun seçileceği, problem türüne ve ağın mimarisine bağlıdır. ReLU genellikle en yaygın kullanılan fonksiyon olsa da, belirli durumlarda Sigmoid, Tanh veya Softmax tercih edilebilir.

## Daha Fazlası İçin Kaynaklar
- [**"Deep Learning"** (Ian Goodfellow and Yoshua Bengio and Aaron Courville)](https://www.deeplearningbook.org/)
- [**"Neural Networks and Deep Learning"** (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/)