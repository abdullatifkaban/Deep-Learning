# Transfer Learning (Transfer Öğrenme)

## Giriş

Transfer learning, önceden eğitilmiş bir modelin bilgisini yeni bir probleme aktarma tekniğidir. Bu bölümde, TensorFlow ile transfer learning uygulamalarını öğreneceğiz.

## Transfer Learning Türleri

### 1. Feature Extraction
```python
import tensorflow as tf

# Pre-trained model yükleme
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Base model'i dondur
base_model.trainable = False

# Yeni model oluştur
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

### 2. Fine-tuning
```python
# Son katmanları eğitilebilir yap
base_model.trainable = True

# Belirli katmanlardan sonrasını eğit
for layer in base_model.layers[:-4]:
    layer.trainable = False

# İki aşamalı eğitim
def two_phase_training(model, train_data, val_data):
    # Faz 1: Feature extraction
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )
    
    # Faz 2: Fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )
    
    return history1, history2
```

## Popüler Pre-trained Modeller

### 1. ResNet
```python
# ResNet50 kullanımı
resnet_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Özelleştirilmiş model
model = tf.keras.Sequential([
    resnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes)
])
```

### 2. EfficientNet
```python
# EfficientNetB0 kullanımı
efficient_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Özelleştirilmiş model
model = tf.keras.Sequential([
    efficient_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes)
])
```

## Özel Uygulamalar

### 1. Görüntü Sınıflandırma
```python
def build_transfer_classifier(base_model_name='ResNet50', num_classes=10):
    # Base model seçimi
    if base_model_name == 'ResNet50':
        base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    elif base_model_name == 'EfficientNetB0':
        base = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)
    
    # Model oluşturma
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
```

### 2. Nesne Tespiti
```python
# SSD MobileNet kullanımı
def build_transfer_detector():
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(300, 300, 3)
    )
    
    # Özellik haritaları
    c3 = base_model.get_layer('block_6_expand_relu').output
    c4 = base_model.get_layer('block_13_expand_relu').output
    c5 = base_model.get_layer('Conv_1').output
    
    # SSD başlıkları
    outputs = []
    for feature_map in [c3, c4, c5]:
        outputs.extend([
            tf.keras.layers.Conv2D(num_classes, 3, padding='same')(feature_map),
            tf.keras.layers.Conv2D(4, 3, padding='same')(feature_map)
        ])
    
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)
```

## Model Hub Kullanımı

### 1. TensorFlow Hub
```python
import tensorflow_hub as hub

# BERT modeli yükleme
bert_model = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    trainable=True
)

# Özelleştirilmiş model
def build_classifier(bert_model, num_classes):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessed = bert_preprocess(text_input)
    outputs = bert_model(preprocessed)
    
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation='softmax')(net)
    
    return tf.keras.Model(text_input, net)
```

## Alıştırmalar

1. Görüntü Sınıflandırma:
   - Kendi veri setinizi hazırlayın
   - Farklı pre-trained modelleri deneyin
   - Fine-tuning stratejilerini karşılaştırın

2. Nesne Tespiti:
   - SSD MobileNet modelini kullanın
   - Kendi nesne sınıflarınız için fine-tuning yapın
   - mAP skorunu hesaplayın

3. Metin Sınıflandırma:
   - BERT modelini kullanın
   - Türkçe metin verisi için fine-tuning yapın
   - Farklı öğrenme oranlarını deneyin

## Kaynaklar
1. [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
2. [TensorFlow Hub](https://tfhub.dev/)
3. [Transfer Learning Paper](https://arxiv.org/abs/1911.02685) 