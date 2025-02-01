# Evrişimli Sinir Ağları (Convolutional Neural Networks - CNN)

## 🎯 Hedefler
- CNN'in temel yapısını ve çalışma prensibini anlama
- Evrişim ve havuzlama işlemlerinin mantığını kavrama
- Basit bir CNN modeli oluşturabilme

## 📑 Ön Koşullar
- Python programlama (orta seviye)
- NumPy kütüphanesi kullanımı
- Temel matris işlemleri
- Yapay sinir ağları temelleri

## 🔑 Temel Kavramlar
1. Evrişim (Convolution) İşlemi
2. Havuzlama (Pooling)
3. Aktivasyon Fonksiyonları
4. Tam Bağlantılı Katmanlar

## Giriş
> Zorluk Seviyesi: 🟡 Orta

Evrişimli Sinir Ağları, özellikle görüntü işleme ve bilgisayarlı görü problemlerinde kullanılan özel bir yapay sinir ağı türüdür.

### Neden CNN?
- Görüntülerdeki uzamsal ilişkileri yakalayabilme
- Parametre sayısını azaltma
- Özellik çıkarımını otomatikleştirme

![CNN Architecture](https://miro.medium.com/max/2000/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

## CNN'in Temel Bileşenleri

### 1. Evrişim Katmanı (Convolution Layer)
💡 İpucu: Evrişim işlemi, görüntüdeki özellikleri tespit etmek için kullanılır
```python
import tensorflow as tf

# 2D Evrişim katmanı
conv_layer = tf.keras.layers.Conv2D(
    filters=32,        # Filtre sayısı
    kernel_size=3,     # Filtre boyutu
    strides=1,         # Adım sayısı
    padding='same',    # Kenar dolgulama
    input_shape=(28, 28, 1)  # Girdi boyutu
)
```

![Convolution Operation](https://miro.medium.com/max/1400/1*1okwhewf5KCtIPaFib4XaA.gif)

### 2. Havuzlama Katmanı (Pooling Layer)
```python
# Max Pooling
pool_layer = tf.keras.layers.MaxPooling2D(
    pool_size=2,
    strides=2
)
```

![Max Pooling](https://miro.medium.com/max/1400/1*uoWYsCV5vBU8SHFPAPao-w.gif)

### 3. Aktivasyon Fonksiyonu
```python
# ReLU aktivasyonu
activation = tf.keras.layers.ReLU()
```

### 4. Tam Bağlantılı Katman
```python
# Fully Connected Layer
fc_layer = tf.keras.layers.Dense(10)  # Çıktı sınıf sayısı: 10
```

## Örnek CNN Mimarisi

```python
class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Evrişim bloğu 1
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(2, 2)
        ])
        
        # Evrişim bloğu 2
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(2, 2)
        ])
        
        # Tam bağlantılı katmanlar
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ])
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.reshape(x, (x.shape[0], -1))  # Flatten
        x = self.fc(x)
        return x
```

## Evrişim İşlemi Detayları

### 1. Filtre (Kernel)
- Özellik çıkarımı için kullanılan küçük matris
- Görüntü üzerinde kaydırılır

![Kernel Operation](https://miro.medium.com/max/1400/1*GcI7G-JLAQiEoCON7xFbhg.gif)

### 2. Stride ve Padding
```python
# Stride = 2, Padding = 1
conv = tf.keras.layers.Conv2D(1, 32, kernel_size=3, strides=2, padding='same')
```

![Stride and Padding](https://miro.medium.com/max/1400/1*BMngs93_rm2_BpJFH2mS0Q.gif)

## Modern CNN Mimarileri

### 1. VGG16
```python
def vgg16():
    return tf.keras.Sequential([
        # Conv Block 1
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Conv Block 2
        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # FC Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1000)
    ])
```

### 2. ResNet
```python
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        
    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv2(x)
        x += residual  # Skip connection
        x = tf.keras.layers.ReLU()(x)
        return x
```

## Transfer Learning

```python
import tensorflow as tf

# Pre-trained model yükleme
model = tf.keras.applications.ResNet18(weights='imagenet', include_top=False)

# Son katmanı değiştirme
num_features = model.layers[-1].output.shape[-1]
model.layers[-1] = tf.keras.layers.Dense(num_classes, activation='softmax')
```

## Veri Artırma (Data Augmentation)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

## Örnek Uygulamalar

### 3. Style Transfer
```python
def build_style_transfer_model():
    # VGG19 modelini yükle
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Style ve content katmanları
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                   'block4_conv1', 'block5_conv1']
    content_layers = ['block5_conv2']
    
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    
    return tf.keras.Model(vgg.input, style_outputs + content_outputs)
```

### 4. Segmentasyon
```python
def build_unet_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Bridge
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    
    # Decoder
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv2)
    concat1 = tf.keras.layers.Concatenate()([conv1, up1])
    conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)
    conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv3)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### 5. Super Resolution
```python
def build_super_resolution_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 9, padding='same', input_shape=(None, None, 3)),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        
        tf.keras.layers.Conv2D(32, 1, padding='same'),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        
        tf.keras.layers.Conv2D(3, 5, padding='same'),
    ])
    return model
```

### 6. Anomali Tespiti
```python
def build_autoencoder_model(input_shape):
    # Encoder
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # Decoder
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Conv2D(input_shape[-1], 3, activation='sigmoid', padding='same')(x)
    
    return tf.keras.Model(inputs, outputs)
```

## Görselleştirme Örnekleri

### 1. Aktivasyon Haritaları
```python
def visualize_activation_maps(model, img, layer_name):
    # Ara katman çıktısını almak için model oluştur
    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Aktivasyon haritalarını al
    activation = activation_model.predict(img[np.newaxis, ...])
    
    # Görselleştirme
    plt.figure(figsize=(15, 5))
    for i in range(min(activation.shape[-1], 16)):
        plt.subplot(2, 8, i+1)
        plt.imshow(activation[0, :, :, i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i}')
    plt.tight_layout()
    plt.show()
```

### 2. Saliency Maps
```python
def compute_saliency_map(model, img, class_idx):
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img[np.newaxis, ...])
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[:, class_idx]
    
    # Gradyanları hesapla
    grads = tape.gradient(loss, img_tensor)
    grads = tf.abs(grads)
    
    # Normalize et
    grads = tf.reduce_max(grads, axis=-1)
    grads = (grads - tf.reduce_min(grads)) / (tf.reduce_max(grads) - tf.reduce_min(grads))
    
    return grads[0].numpy()
```

### 3. Grad-CAM
```python
def generate_gradcam(model, img, layer_name, class_idx):
    grad_model = tf.keras.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img[np.newaxis, ...])
        loss = predictions[:, class_idx]
    
    # Feature map gradyanları
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Ağırlıklı toplam
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize et
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
```

### 4. Filter Visualization
```python
def visualize_filters(model, layer_name, filter_index):
    layer = model.get_layer(layer_name)
    
    # Gradyan yükselişi ile filtre görselleştirme
    input_img = tf.random.uniform((1, 128, 128, 3))
    iterations = 30
    learning_rate = 10.
    
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(input_img)
            activation = layer(input_img)
            loss = tf.reduce_mean(activation[..., filter_index])
        
        # Gradyanları hesapla ve güncelle
        grads = tape.gradient(loss, input_img)
        grads = tf.math.l2_normalize(grads)
        input_img += learning_rate * grads
    
    # Normalize et
    img = input_img[0].numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    return img
```

### 5. t-SNE Görselleştirme
```python
def visualize_embeddings(model, data, labels, layer_name):
    # Feature extraction modeli
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Feature'ları çıkar
    features = feature_model.predict(data)
    features_flat = features.reshape(len(features), -1)
    
    # t-SNE uygula
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features_flat)
    
    # Görselleştir
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f't-SNE visualization of {layer_name} features')
    plt.show()
```

## 📚 Önerilen Kaynaklar
- [CS231n CNN Dersleri](http://cs231n.github.io/)
- [TensorFlow CNN Öğreticisi](https://www.tensorflow.org/tutorials/images/cnn)
- [CNN Görselleştirme](https://poloclub.github.io/cnn-explainer/)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. MNIST veri seti üzerinde basit CNN oluşturma
2. Farklı filtre boyutlarını deneme

### Orta Seviye
1. Veri artırma teknikleri uygulama
2. Transfer learning ile model geliştirme

### İleri Seviye
1. Özel veri seti üzerinde model eğitme
2. Model optimizasyonu yapma

## Kaynaklar
1. [CS231n - Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
2. [Deep Learning Book - CNN Chapter](https://www.deeplearningbook.org/contents/convnets.html)
3. [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn) 