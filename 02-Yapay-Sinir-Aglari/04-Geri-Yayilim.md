# Geri Yayılım Algoritması

## 📍 Bölüm Haritası
- Önceki Bölüm: [Aktivasyon Fonksiyonları](03-Aktivasyon-Fonksiyonlari.md)
- Sonraki Bölüm: [CNN Mimarisi](../03-Derin-Ogrenme-Modelleri/01-CNN.md)
- Tahmini Süre: 4-5 saat
- Zorluk Seviyesi: 🟡 Orta

## 🎯 Hedefler
- Geri yayılım algoritmasının mantığını anlama
- Gradyan hesaplamalarını kavrama
- Zincir kuralını uygulama
- Optimizasyon tekniklerini öğrenme

## 🎯 Öz Değerlendirme
- [ ] Geri yayılım algoritmasını açıklayabiliyorum
- [ ] Gradyan hesaplamalarını yapabiliyorum
- [ ] Zincir kuralını uygulayabiliyorum
- [ ] Farklı optimizasyon tekniklerini kullanabiliyorum

## 🚀 Mini Projeler
1. Manuel Geri Yayılım
   - Basit sinir ağı implementasyonu
   - Gradyan hesaplamaları
   - Ağırlık güncellemeleri

2. Optimizasyon Karşılaştırması
   - SGD vs Adam
   - Momentum implementasyonu
   - Öğrenme oranı planlaması

## 📑 Ön Koşullar
- Türev ve zincir kuralı bilgisi
- Python ve NumPy deneyimi
- Temel lineer cebir
- Optimizasyon kavramları

## 🔑 Temel Kavramlar
1. Gradyan İniş
2. Zincir Kuralı
3. Hata Fonksiyonları
4. Optimizasyon Teknikleri

## Giriş

Geri yayılım algoritması, yapay sinir ağlarının eğitiminde kullanılan temel optimizasyon yöntemidir. Bu algoritma, ağın çıktısı ile beklenen çıktı arasındaki hatayı geriye doğru yayarak ağırlıkları günceller.

![Backpropagation](https://upload.wikimedia.org/wikipedia/commons/4/4f/Backpropagation_Algorithm.gif)

## Algoritmanın Adımları

### 1. İleri Yayılım (Forward Propagation)
```python
import tensorflow as tf

class SimpleNN(tf.keras.Model):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
```

### 2. Hata Hesaplama
```python
# Kategorik çapraz entropi kaybı
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# MSE kaybı
loss_fn = tf.keras.losses.MeanSquaredError()
```

### 3. Gradyan Hesaplama ve Güncelleme
```python
@tf.function
def train_step(model, inputs, targets, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    
    # Gradyanları hesapla
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Ağırlıkları güncelle
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
```

## Tam Implementasyon

```python
class NeuralNetwork(tf.keras.Model):
    def __init__(self, layer_sizes):
        super(NeuralNetwork, self).__init__()
        self.layers_list = []
        
        for size in layer_sizes[:-1]:
            self.layers_list.append(tf.keras.layers.Dense(size, activation='relu'))
        self.layers_list.append(tf.keras.layers.Dense(layer_sizes[-1]))
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x

# Model eğitimi
@tf.function
def train_model(model, dataset, optimizer, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataset:
            loss = train_step(model, batch_x, batch_y, optimizer)
            total_loss += loss
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")
```

## Optimizasyon Teknikleri

### 1. Momentum
```python
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9
)
```

### 2. Adam Optimizer
```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)
```

## Örnek: MNIST Sınıflandırma

```python
# Veri yükleme
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veri ön işleme
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Dataset oluşturma
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

# Model oluşturma
model = NeuralNetwork([784, 128, 64, 10])

# Optimizer ve loss fonksiyonu
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Model derleme
model.compile(optimizer=optimizer,
             loss=loss_fn,
             metrics=['accuracy'])

# Model eğitimi
history = model.fit(train_dataset,
                   epochs=10,
                   validation_data=(x_test, y_test))
```

## Gradyan Görselleştirme

```python
# Gradyanları izleme
@tf.function
def get_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_fn(targets, predictions)
    return tape.gradient(loss, model.trainable_variables)

# Gradyanları görselleştirme
def plot_gradients(gradients):
    plt.figure(figsize=(10, 6))
    for i, grad in enumerate(gradients):
        plt.subplot(len(gradients), 1, i+1)
        plt.hist(grad.numpy().flatten(), bins=50)
        plt.title(f'Layer {i+1} Gradients')
    plt.tight_layout()
    plt.show()
```

## Vanishing ve Exploding Gradients

### Vanishing Gradients
- Derin ağlarda gradyanların çok küçülmesi
- Çözüm: ReLU aktivasyonu, LSTM, ResNet

### Exploding Gradients
- Gradyanların çok büyümesi
- Çözüm: Gradient Clipping, Weight Initialization

## Alıştırmalar

1. XOR problemi için:
   - 2-2-1 mimarisinde bir ağ oluşturun
   - Geri yayılım algoritmasını implement edin
   - Farklı öğrenme oranlarını deneyin

2. MNIST veri seti için:
   - Çok katmanlı bir ağ oluşturun
   - Momentum ve Adam optimizerları karşılaştırın
   - Öğrenme eğrilerini çizin

3. Vanishing Gradient problemi:
   - Derin bir ağ oluşturun
   - Sigmoid ve ReLU aktivasyonlarını karşılaştırın
   - Gradyanları görselleştirin

## Kaynaklar
1. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
2. [TensorFlow Tutorial - Custom Training](https://www.tensorflow.org/tutorials/customization/custom_training)
3. [Deep Learning Book - Optimization](https://www.deeplearningbook.org/contents/optimization.html) 