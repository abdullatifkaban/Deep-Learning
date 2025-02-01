# Perceptron ve Sinir Hücreleri

## Biyolojik Sinir Hücresi

Yapay sinir ağları, biyolojik sinir sisteminden esinlenilerek geliştirilmiştir.

![Biyolojik Nöron](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Neuron.svg/1200px-Neuron.svg.png)

### Biyolojik Nöronun Bileşenleri
1. **Dendritler**: Diğer nöronlardan gelen sinyalleri alır
2. **Hücre Gövdesi (Soma)**: Sinyalleri işler
3. **Akson**: İşlenmiş sinyali diğer nöronlara iletir
4. **Sinaps**: Nöronlar arası bağlantı noktaları

## Yapay Sinir Hücresi (Perceptron)

Perceptron, yapay sinir ağlarının en temel yapı taşıdır.

![Perceptron Yapısı](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Perceptron_moj.png/1200px-Perceptron_moj.png)

### Perceptron'un Bileşenleri
1. **Girdiler (x₁, x₂, ..., xₙ)**: Nöronun aldığı veriler
2. **Ağırlıklar (w₁, w₂, ..., wₙ)**: Her girdinin önem derecesi
3. **Bias (b)**: Eşik değeri
4. **Toplama Fonksiyonu**: Ağırlıklı toplamı hesaplar
5. **Aktivasyon Fonksiyonu**: Çıktıyı belirler

### Matematiksel Model
```python
import tensorflow as tf
import numpy as np

class Perceptron(tf.keras.Model):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.dense = tf.keras.layers.Dense(1, use_bias=True)
        
    def call(self, x):
        # Ağırlıklı toplam ve bias
        z = self.dense(x)
        # Step fonksiyonu
        return tf.cast(tf.greater(z, 0), tf.float32)
```

## Öğrenme Süreci

### 1. İleri Yayılım (Forward Propagation)
```python
@tf.function
def predict(self, x):
    return self.call(x)
```

### 2. Hata Hesaplama
```python
@tf.function
def calculate_error(y_true, y_pred):
    return tf.cast(y_true, tf.float32) - y_pred
```

### 3. Ağırlık Güncelleme
```python
@tf.function
def train_step(self, x, y, learning_rate=0.1):
    with tf.GradientTape() as tape:
        predictions = self.call(x)
        error = self.calculate_error(y, predictions)
    
    gradients = tape.gradient(error, self.trainable_variables)
    for grad, var in zip(gradients, self.trainable_variables):
        var.assign_add(learning_rate * grad)
```

## Örnek: AND Kapısı Uygulaması

```python
# Eğitim verisi
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([0, 0, 0, 1], dtype=np.float32)

# Model oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='step', input_shape=(2,))
])

# Model derleme
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Model eğitimi
history = model.fit(X, y, epochs=100, verbose=0)

# Test
predictions = model.predict(X)
for x_i, pred in zip(X, predictions):
    print(f"Girdi: {x_i}, Çıktı: {int(pred[0])}")
```

## Perceptron'un Sınırlamaları

1. **Doğrusal Ayrılabilirlik**
   - Sadece doğrusal olarak ayrılabilen problemleri çözebilir
   - XOR problemi gibi doğrusal olmayan problemleri çözemez

![Doğrusal Ayrılabilirlik](https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Linearly_separable_data.png/1200px-Linearly_separable_data.png)

2. **Tek Katman**
   - Karmaşık örüntüleri öğrenemez
   - Çok katmanlı problemler için yetersiz

## Çok Katmanlı Algılayıcıya Geçiş

Perceptron'un sınırlamalarını aşmak için:
1. Çok katmanlı yapı
2. Farklı aktivasyon fonksiyonları
3. Geri yayılım algoritması

## Alıştırmalar

1. OR kapısı için perceptron implementasyonu:
```python
# OR kapısı verisi
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([0, 1, 1, 1], dtype=np.float32)

# Model oluşturma ve eğitim
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='step', input_shape=(2,))
])
model.compile(optimizer='sgd', loss='binary_crossentropy')
model.fit(X, y, epochs=100, verbose=0)
```

2. NOT kapısı için perceptron implementasyonu:
```python
# NOT kapısı verisi
X = np.array([[0], [1]], dtype=np.float32)
y = np.array([1, 0], dtype=np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='step', input_shape=(1,))
])
model.compile(optimizer='sgd', loss='binary_crossentropy')
model.fit(X, y, epochs=100, verbose=0)
```

## Kaynaklar
1. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
2. [Deep Learning Book - Chapter 6](https://www.deeplearningbook.org/contents/mlp.html)
3. [TensorFlow Documentation](https://www.tensorflow.org/guide/keras/custom_layers_and_models) 