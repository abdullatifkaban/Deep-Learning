# Aktivasyon Fonksiyonları

## �� Bölüm Haritası
- Önceki Bölüm: [Çok Katmanlı Ağlar](02-Cok-Katmanli-Aglar.md)
- Sonraki Bölüm: [Geri Yayılım](04-Geri-Yayilim.md)
- Tahmini Süre: 2-3 saat
- Zorluk Seviyesi: 🟢 Başlangıç

## 🎯 Hedefler
- Aktivasyon fonksiyonlarının amacını anlama
- Farklı fonksiyonların özelliklerini öğrenme
- Kullanım alanlarını ve seçim kriterlerini kavrama
- Türevlerini ve gradyan hesaplamalarını anlama

## 🎯 Öz Değerlendirme
- [ ] Aktivasyon fonksiyonlarının amacını açıklayabiliyorum
- [ ] Farklı fonksiyonların avantaj/dezavantajlarını biliyorum
- [ ] Uygun fonksiyonu seçebiliyorum
- [ ] Türev hesaplamalarını yapabiliyorum

## 🚀 Mini Projeler
1. Aktivasyon Fonksiyonu Karşılaştırması
   - Farklı fonksiyonları görselleştirme
   - Gradyan değişimlerini analiz etme
   - Performans karşılaştırması

2. Vanishing Gradient Problemi
   - Derin ağlarda problem tespiti
   - Çözüm yöntemleri uygulama
   - Sonuçları değerlendirme

## 📑 Ön Koşullar
...

## Aktivasyon Fonksiyonlarının Önemi

Aktivasyon fonksiyonları, yapay sinir ağlarında doğrusal olmayan ilişkileri modellemek için kullanılan matematiksel fonksiyonlardır. Bu fonksiyonlar, nöronların çıktılarını belirli bir aralığa sıkıştırır ve ağın öğrenme kapasitesini artırır.

## Temel Aktivasyon Fonksiyonları

### 1. Sigmoid Fonksiyonu

$$ f(x) = \frac{1}{1 + e^{-x}} $$

```python
import tensorflow as tf

def sigmoid(x):
    return tf.math.sigmoid(x)

# Keras ile kullanım
activation = tf.keras.layers.Activation('sigmoid')
# veya
layer = tf.keras.layers.Dense(units=10, activation='sigmoid')
```

![Sigmoid](https://miro.medium.com/max/1400/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

**Özellikleri:**
- Çıktı aralığı: (0,1)
- Sürekli ve türevlenebilir
- Gradyan kaybı problemi yaşanabilir
- İkili sınıflandırma problemlerinde çıkış katmanında kullanılır

### 2. ReLU (Rectified Linear Unit)

$$ f(x) = \max(0, x) $$

```python
def relu(x):
    return tf.nn.relu(x)

# Keras ile kullanım
activation = tf.keras.layers.ReLU()
# veya
layer = tf.keras.layers.Dense(units=10, activation='relu')
```

![ReLU](https://miro.medium.com/max/1400/1*XxxiA0jJvPrHEJHD4z893g.png)

**Özellikleri:**
- Hesaplama açısından verimli
- Gradyan kaybı problemini azaltır
- Ölü ReLU problemi yaşanabilir
- En yaygın kullanılan aktivasyon fonksiyonu

### 3. Tanh (Hiperbolik Tanjant)

$$ f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

```python
def tanh(x):
    return tf.math.tanh(x)

# Keras ile kullanım
activation = tf.keras.layers.Activation('tanh')
# veya
layer = tf.keras.layers.Dense(units=10, activation='tanh')
```

![Tanh](https://miro.medium.com/max/1400/1*f9erByySVjTjohfFdNkJYQ.png)

**Özellikleri:**
- Çıktı aralığı: (-1,1)
- Sigmoid'e göre daha iyi gradyan akışı
- Sıfır merkezli

### 4. Leaky ReLU

$$ f(x) = \max(\alpha x, x) $$

```python
def leaky_relu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha=alpha)

# Keras ile kullanım
activation = tf.keras.layers.LeakyReLU(alpha=0.01)
```

![Leaky ReLU](https://miro.medium.com/max/1400/1*A_Bzn0CjUgOXtPCJKnKLqA.png)

**Özellikleri:**
- ReLU'nun ölü nöron problemini çözer
- Negatif değerler için küçük bir eğim sağlar
- α parametresi ayarlanabilir

### 5. Softmax

$$ f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}} $$

```python
def softmax(x):
    return tf.nn.softmax(x)

# Keras ile kullanım
activation = tf.keras.layers.Activation('softmax')
# veya
layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')
```

![Softmax](https://miro.medium.com/max/1400/1*670CdxchunD-yAuUWdI7Bg.png)

**Özellikleri:**
- Çok sınıflı sınıflandırma için kullanılır
- Çıktıların toplamı 1'dir (olasılık dağılımı)
- Genellikle son katmanda kullanılır

## Aktivasyon Fonksiyonu Seçimi

### Gizli Katmanlar İçin
- **ReLU**: Varsayılan seçim, çoğu durumda iyi çalışır
- **Leaky ReLU**: ReLU'nun ölü nöron problemi yaşandığında
- **Tanh**: Sıfır merkezli girdi gereken durumlarda

### Çıkış Katmanı İçin
- **Sigmoid**: İkili sınıflandırma
- **Softmax**: Çok sınıflı sınıflandırma
- **Linear**: Regresyon problemleri

## PyTorch ile Uygulama -> TensorFlow ile Uygulama

```python
# Model tanımlama
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Model derleme
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## TensorFlow ile Örnek Uygulama

```python
# Custom aktivasyon fonksiyonu
class CustomReLU(tf.keras.layers.Layer):
    def __init__(self, threshold=0.0):
        super(CustomReLU, self).__init__()
        self.threshold = threshold
    
    def call(self, inputs):
        return tf.maximum(self.threshold, inputs)

# Model oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    CustomReLU(threshold=0.1),
    tf.keras.layers.Dense(32),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Alıştırmalar

1. Farklı aktivasyon fonksiyonlarını numpy ile implement edin:
   - Sigmoid ve türevi
   - ReLU ve türevi
   - Tanh ve türevi

2. MNIST veri seti üzerinde farklı aktivasyon fonksiyonlarını karşılaştırın:
   - ReLU vs Tanh
   - Sigmoid vs Softmax (çıkış katmanında)
   - Leaky ReLU vs ReLU

3. Gradyan kaybı problemini gözlemleyin:
   - Derin bir ağ oluşturun
   - Sigmoid vs ReLU karşılaştırması yapın
   - Gradyanları görselleştirin

## Kaynaklar
1. [CS231n - Neural Networks](http://cs231n.github.io/neural-networks-1/)
2. [Deep Learning Book - Activation Functions](https://www.deeplearningbook.org/contents/mlp.html)
3. [TensorFlow Activation Functions](https://www.tensorflow.org/api_docs/python/tf/keras/activations) 