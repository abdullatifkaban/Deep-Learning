# Temel Yapay Sinir Ağları

Burada, yapay sinir ağlarının temellerini basit bir örnek üzerinden inceleyeceğiz. Yapay sinir ağları, insan beyninin yapısından esinlenerek oluşturulmuş hesaplama modelleridir ve verilerdeki desenleri öğrenmek için kullanılır. Bu örnekte, Keras kullanarak basit bir feedforward (ileri beslemeli) sinir ağı oluşturup eğiteceğiz.

---

## 1. Giriş

Yapay sinir ağları (YSA), nöron adı verilen temel hesaplama birimlerinin birbirine bağlanmasıyla oluşur. En basit haliyle, **feedforward sinir ağı**, bilgilerin yalnızca giriş katmanından çıkış katmanına doğru aktığı yapıdır. Bu örnekte, XOR problemi gibi basit bir ikili sınıflandırma problemi üzerinde çalışacağız:

- **Giriş (x):** İkili çiftlerden oluşan veri seti.
- **Çıkış (y):** Her bir çift için beklenen sonuç.

---

## 2. Kod Örneği

Aşağıda, basit bir yapay sinir ağı modelini oluşturup eğitmek için kullanılan kod örneği yer almaktadır:

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR problemi için giriş ve beklenen çıkış değerleri
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# Model oluşturma
model = Sequential()
model.add(Dense(2, activation='relu'))  # Gizli katman: 2 nöron ve ReLU aktivasyon fonksiyonu
model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı: 1 nöron ve sigmoid aktivasyon fonksiyonu

# Modeli derleme: Adam optimizörü ve binary crossentropy kayıp fonksiyonu kullanılıyor
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model eğitimi: 100 epoch boyunca model eğitiliyor
history = model.fit(np.array(x), np.array(y), epochs=100, verbose=0)
```

## 3. Model Mimarisi

### Sequential Model
Keras'ın `Sequential` modeli, katmanların sıralı bir şekilde istendiği gibi oluşturulmasını sağlar.

### Gizli Katman
İlk katman, 2 nöron içerir ve `ReLU` (Rectified Linear Unit) aktivasyon fonksiyonu kullanır.
- **ReLU:** Giriş değeri pozitifse onu geçirir, negatifse 0 değeri üretir. Bu özellik, modelin doğrusal olmayan ilişkileri öğrenmesini sağlar.

### Çıkış Katmanı
Son katman, tek bir nörondan oluşur ve sigmoid aktivasyon fonksiyonu kullanır.
- **Sigmoid:** Çıktıyı 0 ile 1 arasında sınırlar ve ikili sınıflandırma problemlerinde idealdir.

## 4. Model Derleme ve Eğitimi

### Derleme
- **Optimizer (Adam):** Ağırlıkların hızlı ve etkili bir şekilde güncellenmesini sağlayan optimizasyon algoritmasıdır.
- **Kayıp Fonksiyonu (Binary Crossentropy):** İkili sınıflandırma problemlerinde modelin tahminleri ile gerçek değerler arasındaki farkı ölçer.

### Eğitim
Model, 100 epoch boyunca eğitim verisi üzerinde eğitilir. Bu süreçte, model ağırlıklarını güncelleyerek kayıp fonksiyonunu minimize etmeye çalışır.


### Test
Şimdi yukarıda oluşturduğumuz modeli test edelim.

Modelin performansını değerlendirmek için yine ` x ` verilerini kullanacağız ve sonuçları analiz edeceğiz.

```python
model.predict(np.array(x))
```
### Sonuç

Bu örnekte, basit bir yapay sinir ağı modeli oluşturup XOR problemi üzerinde eğittik. Modelimiz, iki gizli nöron ve bir çıkış nöronundan oluşan bir `Sequential` modeldir. ReLU ve sigmoid aktivasyon fonksiyonları kullanılarak modelin doğrusal olmayan ilişkileri öğrenmesi sağlanmıştır. Adam optimizörü ve binary crossentropy kayıp fonksiyonu ile modelimizi derledik ve 100 epoch boyunca eğittik. Sonuç olarak, modelimizin performansını test ederek XOR problemini başarıyla çözdüğünü gözlemledik.

Bu örnekte hem teorik hem de pratik olarak aşağıdaki temeller ele alındı:

* Veri hazırlığı,
* Model mimarisi,
* Model derleme ve eğitimi,
* Öğrenme sürecinin temel kavramları

Bu temeller, daha karmaşık yapay sinir ağı modelleri geliştirirken sağlam bir temel oluşturur. Kendi problemleriniz üzerinde denemeler yaparak bu modeli genişletebilir ve farklı uygulamalar geliştirebilirsiniz.

