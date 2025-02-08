# Aktivasyon Fonksiyonları

## 1. Aktivasyon Fonksiyonlarının Önemi

Aktivasyon fonksiyonları, yapay sinir ağlarında her nöronun çıkışını belirleyen matematiksel fonksiyonlardır. Temel olarak, girdileri alır ve belirli bir aralıkta çıktı üretirler. Aktivasyon fonksiyonları, ağın öğrenme kapasitesini ve doğruluğunu doğrudan etkileyen önemli bileşenlerdir.

## 2. Yaygın Aktivasyon Fonksiyonları

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

[Kaynak](https://miro.medium.com/max/1400/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

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

[Kaynak](https://miro.medium.com/max/1400/1*XxxiA0jJvPrHEJHD4z893g.png)

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

![Tanh](https://www.bilgisayarkavramlari.com/wp-content/uploads/2008/10/tanh_plot.gif)

[Kaynak](https://www.bilgisayarkavramlari.com/wp-content/uploads/2008/10/tanh_plot.gif)

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

![Leaky ReLU](https://www.researchgate.net/publication/358306930/figure/fig2/AS:1119417702318091@1643901386378/ReLU-activation-function-vs-LeakyReLU-activation-function.png)

[Kaynak](https://www.researchgate.net/publication/358306930/figure/fig2/AS:1119417702318091@1643901386378/ReLU-activation-function-vs-LeakyReLU-activation-function.png)

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

![Softmax](https://cdn.botpenguin.com/assets/website/Softmax_Function_07fe934386.png)

[Kaynak](https://cdn.botpenguin.com/assets/website/Softmax_Function_07fe934386.png)

**Özellikleri:**
- Çok sınıflı sınıflandırma için kullanılır
- Çıktıların toplamı 1'dir (olasılık dağılımı)
- Genellikle son katmanda kullanılır

## 3. Sonuç
Aktivasyon fonksiyonları, yapay sinir ağlarının başarısını doğrudan etkileyen temel bileşenlerdir. Hangi fonksiyonun seçileceği, problem türüne ve ağın mimarisine bağlıdır. ReLU genellikle en yaygın kullanılan fonksiyon olsa da, belirli durumlarda Sigmoid, Tanh veya Softmax tercih edilebilir.

**Önerilen Kaynaklar:**
- "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Neural Networks and Deep Learning" - Michael Nielsen
