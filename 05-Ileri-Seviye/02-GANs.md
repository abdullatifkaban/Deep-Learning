# Üretici Çekişmeli Ağlar (GANs)

## 📍 Bölüm Haritası
- Önceki Bölüm: [Transfer Learning](01-Transfer-Learning.md)
- Sonraki Bölüm: [Reinforcement Learning](03-Reinforcement-Learning.md)
- Tahmini Süre: 6-7 saat
- Zorluk Seviyesi: 🔴 İleri

## 🎯 Hedefler
- GAN mimarisini detaylı anlama
- Generator ve Discriminator yapılarını kavrama
- Farklı GAN türlerini öğrenme
- Modern GAN uygulamaları geliştirme

## 🎯 Öz Değerlendirme
- [ ] GAN mimarisini açıklayabiliyorum
- [ ] Generator ve Discriminator yapılarını anlayabiliyorum
- [ ] Farklı GAN türlerini kullanabiliyorum
- [ ] Mode collapse problemini çözebiliyorum

## 🚀 Mini Projeler
1. DCGAN
   - Yüz üretimi
   - Style transfer
   - Latent space analizi

2. Conditional GAN
   - Etiket tabanlı üretim
   - Cross-domain transfer
   - Attribute manipulation

## 📑 Ön Koşullar
- Derin öğrenme temelleri
- CNN mimarisi bilgisi
- Python ve framework deneyimi
- Optimizasyon teknikleri

## 🔑 Temel Kavramlar
1. Generator
2. Discriminator
3. Adversarial Training
4. Mode Collapse

## Giriş

GANs (Generative Adversarial Networks), üretici ve ayırt edici olmak üzere iki ağın birbirleriyle rekabet ederek öğrendiği derin öğrenme modelidir.

## Temel GAN Mimarisi

### 1. Generator (Üretici)
```python
import tensorflow as tf

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        # Başlangıç katmanı
        tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((7, 7, 256)),
        
        # Upsampling katmanları
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        
        # Çıkış katmanı
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model
```

### 2. Discriminator (Ayırt Edici)
```python
def build_discriminator():
    model = tf.keras.Sequential([
        # Giriş katmanı
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        
        # Özellik çıkarma katmanları
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        
        # Çıkış katmanı
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model
```

## GAN Eğitimi

### 1. Loss Fonksiyonları
```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

### 2. Eğitim Adımı
```python
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator çıktısı
        generated_images = generator(noise, training=True)
        
        # Discriminator tahminleri
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        # Loss hesaplama
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Gradyanları hesapla
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Optimize et
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
```

## GAN Türleri

### 1. DCGAN (Deep Convolutional GAN)
```python
def build_dcgan_generator(latent_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8*8*1024, input_shape=(latent_dim,)),
        tf.keras.layers.Reshape((8, 8, 1024)),
        tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')
    ])
    return model
```

### 2. Conditional GAN
```python
def build_conditional_generator(latent_dim, num_classes):
    # Gürültü girişi
    noise = tf.keras.layers.Input(shape=(latent_dim,))
    
    # Sınıf etiketi girişi
    label = tf.keras.layers.Input(shape=(1,))
    label_embedding = tf.keras.layers.Embedding(num_classes, 50)(label)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)
    
    # Gürültü ve etiketi birleştir
    combined_input = tf.keras.layers.Concatenate()([noise, label_embedding])
    
    # Generator ağı
    x = tf.keras.layers.Dense(7*7*256)(combined_input)
    x = tf.keras.layers.Reshape((7, 7, 256))(x)
    x = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    img = tf.keras.layers.Conv2D(1, 5, padding='same', activation='tanh')(x)
    
    return tf.keras.Model([noise, label], img)
```

## Örnek Uygulamalar

### 1. MNIST GAN
```python
# Veri yükleme
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# Model oluşturma
generator = build_generator(100)
discriminator = build_discriminator()

# Eğitim
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

for epoch in range(EPOCHS):
    for image_batch in train_dataset:
        train_step(image_batch)
```

### 2. Style Transfer GAN
```python
def build_style_transfer_model():
    # VGG19 tabanlı özellik çıkarıcı
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Style ve content katmanları
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    
    model = tf.keras.Model(vgg.input, style_outputs + content_outputs)
    return model
```

## Alıştırmalar

1. MNIST GAN:
   - Basit GAN modelini implement edin
   - Farklı hiperparametrelerle deneyin
   - Mode collapse problemini gözlemleyin

2. Conditional GAN:
   - MNIST üzerinde cGAN implement edin
   - Belirli rakamlar üretin
   - Sonuçları değerlendirin

3. Style Transfer:
   - Style transfer GAN implement edin
   - Kendi resimlerinizle deneyin
   - Farklı style ağırlıklarını test edin

## Kaynaklar
1. [GAN Paper](https://arxiv.org/abs/1406.2661)
2. [DCGAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
3. [Style Transfer Guide](https://www.tensorflow.org/tutorials/generative/style_transfer) 