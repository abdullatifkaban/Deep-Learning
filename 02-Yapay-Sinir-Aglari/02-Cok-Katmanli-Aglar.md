# Çok Katmanlı Sinir Ağları

## 📍 Bölüm Haritası
- Önceki Bölüm: [Perceptron](01-Perceptron.md)
- Sonraki Bölüm: [Aktivasyon Fonksiyonlari](03-Aktivasyon-Fonksiyonlari.md)
- Tahmini Süre: 4-5 saat
- Zorluk Seviyesi: 🟡 Orta

## 🎯 Hedefler
- Çok katmanlı ağ mimarisini anlama
- İleri ve geri yayılımı kavrama
- Farklı aktivasyon fonksiyonlarını uygulama
- Model optimizasyonu yapabilme

## 🎯 Öz Değerlendirme
- [ ] MLP mimarisini açıklayabiliyorum
- [ ] Geri yayılım algoritmasını anlayabiliyorum
- [ ] Hiperparametre optimizasyonu yapabiliyorum
- [ ] Farklı aktivasyon fonksiyonlarını kullanabiliyorum

## 🚀 Mini Projeler
1. MNIST Sınıflandırma
   - MLP modeli oluşturma
   - Hiperparametre optimizasyonu
   - Model performans analizi

2. Regresyon Problemi
   - Boston Housing veri seti
   - Farklı aktivasyon fonksiyonları
   - Dropout implementasyonu

## 📑 Ön Koşullar

Çok Katmanlı Ağlar (Multilayer Perceptron - MLP), tek katmanlı perceptron'un sınırlamalarını aşmak için geliştirilmiş daha karmaşık yapılardır.

![MLP Yapısı](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/1200px-Colored_neural_network.svg.png)

## Ağ Mimarisi

### 1. Giriş Katmanı (Input Layer)
```python
import tensorflow as tf

# Giriş katmanı
input_layer = tf.keras.layers.Input(shape=(input_size,))
```

### 2. Gizli Katmanlar (Hidden Layers)
```python
# Gizli katmanlar
hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
hidden_layer2 = tf.keras.layers.Dense(32, activation='relu')(hidden_layer1)
```

### 3. Çıkış Katmanı (Output Layer)
```python
# Sınıflandırma için
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(hidden_layer2)

# Regresyon için
output_layer = tf.keras.layers.Dense(1, activation='linear')(hidden_layer2)
```

## İleri Yayılım (Forward Propagation)

Her katmanda gerçekleşen işlemler:
1. Ağırlıklı toplam
2. Aktivasyon fonksiyonu

```python
class MLPModel(tf.keras.Model):
    def __init__(self, layer_sizes):
        super(MLPModel, self).__init__()
        self.layers_list = []
        for size in layer_sizes[:-1]:
            self.layers_list.append(tf.keras.layers.Dense(size, activation='relu'))
        self.layers_list.append(tf.keras.layers.Dense(layer_sizes[-1]))
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x
```

## Hiperparametreler

1. **Katman Sayısı ve Boyutu**
   - Çok az: Yetersiz öğrenme (underfitting)
   - Çok fazla: Aşırı öğrenme (overfitting)

2. **Öğrenme Oranı (Learning Rate)**
   - Çok küçük: Yavaş öğrenme
   - Çok büyük: Yakınsama sorunları
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

3. **Batch Size**
   - Küçük: Daha iyi genelleme
   - Büyük: Daha hızlı eğitim
```python
model.fit(X_train, y_train, batch_size=32, epochs=100)
```

## Örnek: XOR Problemi

```python
# Model oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# XOR verisi
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Model derleme
model.compile(optimizer='adam', 
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Model eğitimi
history = model.fit(X, y, epochs=1000, verbose=0)

# Test
predictions = model.predict(X)
print("XOR Çıktıları:")
for x_i, pred in zip(X, predictions):
    print(f"Girdi: {x_i}, Çıktı: {pred[0]:.2f}")
```

## Düzenlileştirme (Regularization)

### 1. Dropout
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 2. L1/L2 Regularization
```python
regularizer = tf.keras.regularizers.l2(0.01)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, 
                         activation='relu',
                         kernel_regularizer=regularizer),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Model Değerlendirme ve İzleme

```python
# Callback'ler
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

# Model eğitimi
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stopping, checkpoint]
)

# Öğrenme eğrilerini çizme
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## Alıştırmalar

1. MNIST veri seti için MLP modeli oluşturun:
   - En az 2 gizli katman
   - Dropout kullanın
   - Model performansını değerlendirin

2. Farklı hiperparametreleri deneyin:
   - Katman boyutları
   - Öğrenme oranı
   - Batch size

3. Overfitting ve underfitting durumlarını gözlemleyin:
   - Eğitim/validasyon grafiklerini çizin
   - Düzenlileştirme yöntemlerini karşılaştırın

## Kaynaklar
1. [Deep Learning Book - Chapter 6](https://www.deeplearningbook.org/contents/mlp.html)
2. [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
3. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) 