# Görüntü Sınıflandırma

## 🎯 Hedefler
- Temel görüntü sınıflandırma modellerini oluşturma
- Veri ön işleme ve artırma tekniklerini uygulama
- Transfer learning ve fine-tuning yapabilme
- Model performansını değerlendirme ve iyileştirme

## 📑 Ön Koşullar
- CNN mimarisi ve çalışma prensibi
- Python ve TensorFlow/Keras kullanımı
- Temel görüntü işleme bilgisi
- Veri manipülasyonu deneyimi

## 🔑 Temel Kavramlar
1. Veri Ön İşleme
2. Data Augmentation
3. Transfer Learning
4. Fine-tuning
5. Model Değerlendirme

## Veri Hazırlama
> Zorluk Seviyesi: 🟡 Orta

💡 İpucu: Veri ön işleme ve artırma, model performansını önemli ölçüde etkiler

### 1. Veri Yükleme ve Ön İşleme
```python
import tensorflow as tf
import numpy as np

# CIFAR-10 veri setini yükleme
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Veri normalizasyonu
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

### 2. Veri Artırma (Data Augmentation)
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Veri artırma örneği
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(x_train[0:1])
    plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
plt.show()
```

## Model Oluşturma

### 1. Basit CNN Modeli
```python
def build_simple_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

### 2. Transfer Learning ile Model
```python
def build_transfer_model():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 32, 3)
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

## Model Eğitimi

### 1. Eğitim Konfigürasyonu
```python
# Model derleme
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback'ler
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2
    )
]
```

### 2. Eğitim Süreci
```python
# Model eğitimi
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)

# Eğitim grafiklerini çizme
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.show()
```

## Model Değerlendirme

### 1. Test Seti Üzerinde Değerlendirme
```python
# Model değerlendirme
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()
```

### 2. Görsel Sonuçlar
```python
def plot_predictions(model, images, labels, n=5):
    predictions = model.predict(images[:n])
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        pred_label = np.argmax(predictions[i])
        true_label = np.argmax(labels[i])
        plt.title(f'Pred: {pred_label}\nTrue: {true_label}')
        plt.axis('off')
    plt.show()

plot_predictions(model, x_test, y_test)
```

## Özel Veri Seti ile Çalışma

### 1. Veri Yükleme
```python
# Veri seti oluşturma
data_dir = 'path/to/dataset'
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
```

### 2. Model Fine-tuning
```python
# Transfer learning ve fine-tuning
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False
)

# Base model'i dondur
base_model.trainable = False

# Yeni model oluştur
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, outputs)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
```

## Örnek Uygulamalar

### 3. Çoklu Etiket Sınıflandırma
```python
def build_multilabel_classifier():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_labels, activation='sigmoid')  # Çoklu etiket için sigmoid
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Çoklu etiket kaybı
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model
```

### 4. Fine-Grained Sınıflandırma
```python
def build_fine_grained_classifier():
    # Ana model
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(448, 448, 3)  # Daha yüksek çözünürlük
    )
    
    # Attention mekanizması
    def attention_module(x):
        channels = x.shape[-1]
        # Spatial attention
        spatial = tf.keras.layers.Conv2D(1, 1)(x)
        spatial = tf.keras.layers.Activation('sigmoid')(spatial)
        x_spatial = x * spatial
        
        # Channel attention
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
        avg_pool = tf.keras.layers.Dense(channels//8, activation='relu')(avg_pool)
        max_pool = tf.keras.layers.Dense(channels//8, activation='relu')(max_pool)
        channel = tf.keras.layers.Dense(channels, activation='sigmoid')(avg_pool + max_pool)
        x_channel = x * channel
        
        return x_spatial + x_channel
    
    # Model oluşturma
    inputs = tf.keras.layers.Input(shape=(448, 448, 3))
    x = base_model(inputs)
    x = attention_module(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_fine_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### 5. Metric Learning
```python
def build_siamese_network():
    # Özellik çıkarıcı
    def get_encoder():
        encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu')
        ])
        return encoder
    
    # Giriş görüntüleri
    input_1 = tf.keras.layers.Input(shape=(96, 96, 3))
    input_2 = tf.keras.layers.Input(shape=(96, 96, 3))
    
    # Paylaşılan encoder
    encoder = get_encoder()
    feat_1 = encoder(input_1)
    feat_2 = encoder(input_2)
    
    # L2 mesafesi
    distance = tf.keras.layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True))
    )([feat_1, feat_2])
    
    return tf.keras.Model(inputs=[input_1, input_2], outputs=distance)
```

### 6. Self-Supervised Learning
```python
def build_simclr_model():
    # Base encoder
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )
    
    # Projection head
    projection_head = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(128)
    ])
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomCrop(height=224, width=224),
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ])
    
    # Model oluşturma
    inputs = tf.keras.layers.Input((224, 224, 3))
    augmented = data_augmentation(inputs)
    features = base_model(augmented)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = projection_head(features)
    outputs = tf.nn.l2_normalize(outputs, axis=1)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

## 📚 Önerilen Kaynaklar
- [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Keras Applications Guide](https://keras.io/api/applications/)
- [Data Augmentation Best Practices](https://keras.io/api/layers/preprocessing_layers/)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. MNIST veriseti üzerinde basit CNN modeli
2. Temel veri artırma teknikleri uygulama

### Orta Seviye
1. Transfer learning ile özel veri seti sınıflandırma
2. Model performans optimizasyonu

### İleri Seviye
1. Çoklu etiket sınıflandırma
2. Fine-grained sınıflandırma modeli geliştirme

## Kaynaklar
1. [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
2. [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
3. [Data Augmentation Tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation) 