## 8.3 Örnek Uygulama: Görüntü Renklendirme (Conditional GAN - pix2pix)

Bu bölümde, **Koşullu Çekişmeli Üretici Ağlar (Conditional GANs - cGANs)** kullanarak siyah beyaz (grayscale) görüntülere renk katmayı öğreneceğiz. Bu yaklaşım, özellikle **pix2pix** modeliyle popülerleşen bir **Görüntüden Görüntüye Çeviri (Image-to-Image Translation)** tekniğidir. Modelimiz, bir görüntünün siyah beyaz versiyonunu (L kanalı) **koşul** olarak alıp, renkli versiyonunu (AB kanalları) üretmeye çalışacaktır. Ayırt edici model ise, verilen L kanalına karşılık gelen AB kanallarının gerçek mi yoksa üretilmiş mi olduğunu ayırt etmeye çalışacaktır.

**Genel Akış:**

1.  Gerekli kütüphaneleri yükleyeceğiz (`tensorflow_datasets` dahil).
2.  Renkli çiçek görüntüleri içeren `tf_flowers` veri setini yükleyeceğiz.
3.  Görüntüleri işleyerek girdi (L kanalı) ve hedef (AB kanalları) çiftlerini oluşturacağız (LAB renk uzayı kullanarak).
4.  Üretici (Generator) modelini tanımlayacağız (U-Net mimarisi).
5.  Koşullu Ayırt Edici (Conditional Discriminator) modelini tanımlayacağız (PatchGAN tarzı).
6.  Modeller için kayıp fonksiyonlarını (GAN kaybı + L1 kaybı) ve optimizasyon algoritmalarını belirleyeceğiz.
7.  Eğitim adımını (hem üretici hem ayırt edici için) tanımlayacağız.
8.  Modeli eğiteceğiz ve ilerlemeyi görselleştireceğiz.
9.  Eğitilmiş modeli kullanarak yeni siyah beyaz görüntüleri renklendireceğiz.

---

### Adım 1: Kütüphanelerin Yüklenmesi

Gerekli olan TensorFlow, Keras, TensorFlow Datasets, NumPy, Matplotlib ve Scikit-Image kütüphanelerini projemize dahil ediyoruz.

```python
# TensorFlow ve Keras
import tensorflow as tf
from tensorflow.keras import layers, Model

# Veri Seti için
import tensorflow_datasets as tfds

# Yardımcı Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
from skimage import color # Görüntü işleme ve renk uzayı dönüşümleri için (skimage.transform'a gerek kalmayabilir)
import os
import time
from IPython import display

# Tensorflow'un skimage fonksiyonlarını sarmalaması için
from functools import partial
```

**Açıklama:**
*   `tensorflow`, `keras`: Derin öğrenme modelleri için.
*   `tensorflow_datasets`: Hazır veri setlerine kolay erişim için (`tf_flowers`).
*   `numpy`: Sayısal işlemler.
*   `matplotlib.pyplot`: Görselleştirme.
*   `skimage.color`: Renk uzayı dönüşümleri (RGB <-> LAB) için. `pip install scikit-image` ile yüklenmelidir.
*   `os`, `time`, `IPython.display`: Yardımcı fonksiyonlar.
*   `functools.partial`: `tf.py_function` içinde parametre geçişini kolaylaştırmak için kullanılabilir.

---

### Adım 2: Veri Setinin Yüklenmesi ve Hazırlanması (`tf_flowers`)

Bu örnekte `tensorflow_datasets` kütüphanesinden `tf_flowers` veri setini kullanacağız. Bu veri seti çeşitli çiçeklerin renkli görüntülerini içerir.

Veri setini yükledikten sonra `tf.data` pipeline'ı kullanarak şu adımları uygulayacağız:
1.  Görüntüleri `tf.image` kullanarak uygun bir boyuta (örneğin 128x128) getireceğiz.
2.  Görüntüleri [0, 1] aralığında float değerlere normalize edeceğiz.
3.  `tf.py_function` içinde `skimage.color.rgb2lab` kullanarak RGB'den LAB renk uzayına dönüştüreceğiz.
4.  L kanalını modelin girdisi, A ve B kanallarını ise modelin hedef çıktısı olarak ayıracağız.
5.  L kanalını [0, 1] (veya [-1, 1]), AB kanallarını [-1, 1] aralığına normalize edeceğiz. (L kanalı [0, 100] aralığında, AB [-128, 127] civarında olduğundan uygun ölçekleme yapacağız).

```python
# Parametreler
BUFFER_SIZE = 400 # tf_flowers küçük bir set, tümünü belleğe alıp karıştırabiliriz
BATCH_SIZE = 16   # Daha büyük görüntüler ve modeller için batch boyutunu küçültebiliriz
IMG_WIDTH = 128  # Kullanılacak görüntü genişliği
IMG_HEIGHT = 128 # Kullanılacak görüntü yüksekliği

# tf_flowers veri setini yükle
# Not: İlk yüklemede veri seti indirilecektir.
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)
print("Veri seti bilgisi:", info)
num_examples = info.splits['train'].num_examples
print("Toplam örnek sayısı:", num_examples)

# Scikit-image fonksiyonlarını tf.py_function içinde sarmalamak için yardımcı fonksiyon
def np_rgb_to_lab(rgb_image_np):
    """NumPy RGB görüntüsünü (0-1 aralığında) LAB'a çevirir."""
    # rgb_image_np'nin float32 olduğundan emin olalım
    rgb_image_np = rgb_image_np.astype(np.float32)
    lab_image_np = color.rgb2lab(rgb_image_np)
    return lab_image_np.astype(np.float32)

@tf.function
def tf_rgb_to_lab_wrapper(rgb_tensor):
    """TensorFlow RGB tensorünü LAB tensorüne çevirir."""
    lab_tensor = tf.py_function(func=np_rgb_to_lab,
                                inp=[rgb_tensor],
                                Tout=tf.float32)
    # py_function şekil bilgisini kaybedebilir, manuel olarak ayarlayalım
    lab_tensor.set_shape(rgb_tensor.shape)
    return lab_tensor

# Görüntüleri işleyen ve L, AB kanallarını ayıran fonksiyon
def preprocess_image_train(data):
    # Görüntüyü al (TFDS genellikle 'image' anahtarı altında verir)
    img = data['image']

    # 1. Boyutlandır ve Normalize Et (RGB: 0-255 -> 0-1)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32) / 255.0

    # 2. RGB -> LAB dönüşümü
    lab_img = tf_rgb_to_lab_wrapper(img)

    # 3. L ve AB kanallarını ayır
    # L kanalı [0, 100] aralığındadır. Bunu [-1, 1] aralığına getirelim (pix2pix makalesindeki gibi)
    l_channel = lab_img[:, :, :1] # Kanal boyutunu koru
    l_channel_norm = (l_channel / 50.0) - 1.0 # [0, 100] -> [0, 2] -> [-1, 1]

    # AB kanalları yaklaşık [-128, 127] aralığındadır. Bunu [-1, 1] aralığına getirelim.
    ab_channels = lab_img[:, :, 1:]
    ab_channels_norm = ab_channels / 128.0 # [-128, 127] -> yakl. [-1, 1]

    # Girdi (L kanalı) ve Hedef (AB kanalları) döndür
    return l_channel_norm, ab_channels_norm


# Veri setini oluştur ve işlemleri uygula
train_dataset = dataset.map(preprocess_image_train,
                            num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) # Performans için

print("TFDS Veri seti pipeline hazırlandı.")

# Örnek bir batch alıp şekillerini ve değer aralıklarını kontrol edelim
for example_l, example_ab in train_dataset.take(1):
    print("Örnek L batch şekli:", example_l.shape) # (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1) olmalı
    print("Örnek AB batch şekli:", example_ab.shape) # (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 2) olmalı
    print("Örnek L min/max:", tf.reduce_min(example_l).numpy(), tf.reduce_max(example_l).numpy()) # Yaklaşık [-1, 1] olmalı
    print("Örnek AB min/max:", tf.reduce_min(example_ab).numpy(), tf.reduce_max(example_ab).numpy()) # Yaklaşık [-1, 1] olmalı

    # Örnek bir görüntüyü görselleştirelim (ilk görüntünün L kanalı)
    plt.figure()
    plt.imshow((example_l[0].numpy() + 1.0) / 2.0, cmap='gray') # L'yi [0, 1] yapıp göster
    plt.title("Örnek Girdi L Kanalı (Normalize Edilmiş)")
    plt.axis('off')
    plt.show()
```

**Açıklama:**
*   `tensorflow_datasets` (`tfds`): `tf_flowers` veri setini yüklemek için kullanılır. `with_info=True` ile veri seti hakkında meta veriler de alınır.
*   `IMG_WIDTH`, `IMG_HEIGHT`: Görüntü boyutunu 128x128 olarak güncelledik.
*   `BATCH_SIZE`: Daha büyük görüntüler nedeniyle 16'ya düşürüldü (GPU belleğine bağlı olarak ayarlanabilir).
*   `np_rgb_to_lab`, `tf_rgb_to_lab_wrapper`: `skimage.color.rgb2lab` fonksiyonunu `tf.py_function` içinde kullanabilmek için sarmalayıcı fonksiyonlar. `tf.py_function` NumPy fonksiyonlarının TensorFlow grafiğinde çalıştırılmasına olanak tanır.
*   `preprocess_image_train`: TFDS'den gelen her bir veri örneğini (`data` dictionary) işleyen fonksiyon:
    *   `tf.image.resize`: Görüntüyü hedef boyuta getirir.
    *   Normalizasyon (RGB): Piksel değerlerini [0, 1] aralığına getirir.
    *   `tf_rgb_to_lab_wrapper`: RGB'den LAB'a dönüşümü yapar.
    *   Kanal Ayırma ve Normalizasyon (LAB):
        *   L kanalı `[:, :, :1]` ile alınır (kanal boyutu korunur). [0, 100] aralığındaki L kanalı `(L/50.0) - 1.0` formülüyle [-1, 1] aralığına getirilir. Bu, pix2pix makalesinde kullanılan yaygın bir yöntemdir ve genellikle generator/discriminator için daha stabildir.
        *   AB kanalları `[:, :, 1:]` ile alınır. [-128, 127] aralığındaki AB kanalları `/ 128.0` ile [-1, 1] aralığına getirilir.
    *   Fonksiyon, normalize edilmiş L kanalını (girdi) ve normalize edilmiş AB kanallarını (hedef) döndürür.
*   `tf.data.Dataset` Pipeline:
    *   `.map`: `preprocess_image_train` fonksiyonunu veri setindeki her örneğe uygular. `num_parallel_calls=tf.data.AUTOTUNE` ile işlemi hızlandırır.
    *   `.shuffle`, `.batch`, `.prefetch`: Verimli eğitim için standart adımlar.
*   Kontrol: İlk batch'den örneklerin şekilleri ve değer aralıkları yazdırılarak ön işlemenin doğru çalıştığı teyit edilir. Örnek bir L kanalı da görselleştirilir.

---

### Adım 3: Renklendirme Modelinin Oluşturulması (U-Net Mimarisi)

Görüntü renklendirme için popüler ve etkili bir mimari U-Net'tir. U-Net, bir kodlayıcı (encoder) yolu ve bir kod çözücü (decoder) yolundan oluşur. Kodlayıcı, görüntünün uzamsal boyutunu azaltırken özellik haritalarını çıkarır. Kod çözücü ise bu özellikleri kullanarak görüntünün boyutunu tekrar artırır ve hedef çıktıyı (renk kanalları) üretir. U-Net'in önemli bir özelliği, kodlayıcıdaki katmanlardan kod çözücüdeki karşılık gelen katmanlara "atlama bağlantıları" (skip connections) olmasıdır. Bu bağlantılar, kodlayıcıda yakalanan yüksek çözünürlüklü detayların kod çözücüye aktarılmasına yardımcı olur, bu da daha keskin ve detaylı renkli görüntüler üretilmesini sağlar.

Modelimiz girdi olarak L kanalını (IMG_SIZE, IMG_SIZE, 1) alacak ve çıktı olarak AB kanallarını (IMG_SIZE, IMG_SIZE, 2) üretecektir.

```python
def build_unet_model(img_size):
    inputs = layers.Input(shape=[img_size, img_size, 1]) # Girdi: L kanalı

    # Kodlayıcı (Encoder - Downsampling)
    # Blok 1: 64x64 -> 32x32
    conv1 = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1) # Boyut yarıya iner

    # Blok 2: 32x32 -> 16x16
    conv2 = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Blok 3: 16x16 -> 8x8
    conv3 = layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    # Orta Blok (Bottleneck)
    # 8x8
    conv_mid = layers.Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(pool3)
    conv_mid = layers.Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(conv_mid)

    # Kod Çözücü (Decoder - Upsampling)
    # Blok 4: 8x8 -> 16x16
    up4 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv_mid) # Boyut iki katına çıkar
    skip4 = layers.concatenate([conv3, up4], axis=3) # Atlama bağlantısı (Encoder'dan conv3 ile birleştir)
    conv4 = layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(skip4)
    conv4 = layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(conv4)

    # Blok 5: 16x16 -> 32x32
    up5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    skip5 = layers.concatenate([conv2, up5], axis=3) # Atlama bağlantısı (Encoder'dan conv2 ile birleştir)
    conv5 = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(skip5)
    conv5 = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(conv5)

    # Blok 6: 32x32 -> 64x64
    up6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
    skip6 = layers.concatenate([conv1, up6], axis=3) # Atlama bağlantısı (Encoder'dan conv1 ile birleştir)
    conv6 = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(skip6)
    conv6 = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(conv6)

    # Çıkış Katmanı
    # Hedefimiz AB kanalları (2 kanal) ve değerler [-1, 1] aralığında normalize edildiği için 'tanh' aktivasyonu.
    outputs = layers.Conv2D(2, (1, 1), activation='tanh')(conv6) # 1x1 konvolüsyon ile kanal sayısını 2'ye indir

    model = Model(inputs=inputs, outputs=outputs, name="unet_colorization")
    return model

# Modeli oluştur ve özetini göster
colorization_model = build_unet_model(IMG_SIZE)
colorization_model.summary()

# Test: Rastgele L kanalı ile bir AB kanalı üretelim (eğitimden önce)
# (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1) şeklinde rastgele girdi oluşturalım
dummy_l_channel = tf.random.normal([1, IMG_SIZE, IMG_SIZE, 1])
generated_ab_channels = colorization_model(dummy_l_channel, training=False)
print("Eğitimsiz modelin ürettiği AB kanalları şekli:", generated_ab_channels.shape) # (1, IMG_SIZE, IMG_SIZE, 2) olmalı
```

**Açıklama:**
*   `build_unet_model`: U-Net modelini Keras Functional API kullanarak oluşturan fonksiyon.
*   `layers.Input`: Modelin girdi katmanını tanımlar (L kanalı).
*   **Kodlayıcı (Encoder):**
    *   Her blok genellikle iki `Conv2D` katmanından (ReLU aktivasyonu ile) ve ardından bir `MaxPooling2D` katmanından oluşur.
    *   `Conv2D`: Özellik çıkarır. `padding='same'` ile çıktı boyutu girdiyle aynı kalır (stride 1 ise).
    *   `MaxPooling2D`: Uzamsal boyutu (yükseklik ve genişlik) yarıya indirir, hesaplama yükünü azaltır ve daha global özellikleri yakalamaya yardımcı olur.
*   **Orta Blok (Bottleneck):** Kodlayıcının en derin noktasıdır.
*   **Kod Çözücü (Decoder):**
    *   Her blok bir `Conv2DTranspose` katmanı ile başlar. Bu katman, özellik haritasının uzamsal boyutunu artırır (genellikle `strides=(2, 2)` ile iki katına çıkarır).
    *   `layers.concatenate`: Atlama bağlantısını gerçekleştirir. Kod çözücüdeki büyütülmüş özellik haritası (`upX`) ile kodlayıcıdaki karşılık gelen (aynı boyuttaki) özellik haritasını (`convX`) kanal boyutunda birleştirir. Bu, ince detayların korunmasına yardımcı olur.
    *   Birleştirilmiş özellik haritası, yine iki `Conv2D` katmanından geçirilir.
*   **Çıkış Katmanı:**
    *   Son olarak, `Conv2D` katmanı (1x1 çekirdek boyutuyla) kanal sayısını hedef sayıya (AB için 2) indirir.
    *   `activation='tanh'`: Çıktı değerlerini [-1, 1] aralığına sıkıştırır. Bu, AB kanallarını normalize ettiğimiz aralıkla eşleşir.
*   `Model(inputs=..., outputs=...)`: Girdi ve çıktı katmanlarını belirterek modeli oluşturur.
*   `model.summary()`: Modelin katmanlarını, çıktı şekillerini ve parametre sayılarını gösterir.
*   Test kodu: Eğitimden önce modelin rastgele bir L kanalı girdisiyle doğru şekle sahip bir AB kanalı çıktısı ürettiğini doğrular.

---

### Adım 4: Kayıp Fonksiyonu ve Optimizasyon

Modelin ürettiği renk kanalları (AB) ile gerçek renk kanalları arasındaki farkı ölçmek için bir kayıp fonksiyonuna ve bu kaybı minimize etmek için bir optimizasyon algoritmasına ihtiyacımız var.

Görüntüden görüntüye çeviri problemlerinde, özellikle renklendirme gibi görevlerde, **Ortalama Mutlak Hata (Mean Absolute Error - MAE veya L1 Loss)** sıkça kullanılır. MAE, tahmin edilen ve gerçek piksel değerleri arasındaki mutlak farkların ortalamasını alır. Ortalama Karesel Hata'ya (MSE) göre genellikle daha az bulanık sonuçlar üretebilir.

Optimizasyon algoritması olarak yine **Adam** kullanacağız.

```python
# Kayıp Fonksiyonu: Ortalama Mutlak Hata (MAE)
# Modelin çıktısı (tahmin edilen AB kanalları) ile gerçek AB kanalları arasındaki farkı ölçer.
mae_loss = tf.keras.losses.MeanAbsoluteError()

# Optimizasyon Algoritması
# Adam optimizer genellikle iyi bir başlangıç noktasıdır.
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5) # GAN'lardakine benzer parametreler denenebilir
```

**Açıklama:**
*   `tf.keras.losses.MeanAbsoluteError()`: Tahmin edilen AB kanalları ile hedef (gerçek) AB kanalları arasındaki piksel bazında mutlak farkların ortalamasını hesaplar. Model bu değeri minimize etmeye çalışacaktır.
*   `tf.keras.optimizers.Adam`: Modelin ağırlıklarını, hesaplanan kayba göre güncellemek için kullanılacak optimizasyon algoritması. `learning_rate` ve `beta_1` gibi hiperparametreler eğitimin hızını ve kararlılığını etkiler.

---

### Adım 5: Eğitim Adımının Tanımlanması

Tek bir eğitim adımında modelin nasıl güncelleneceğini tanımlayan fonksiyonu yazıyoruz. Bu fonksiyon, bir batch L kanalı (girdi) ve karşılık gelen AB kanallarını (hedef) alır, modeli çalıştırır, kaybı hesaplar ve modelin ağırlıklarını günceller. `@tf.function` dekoratörü, eğitimi hızlandırmak için bu adımı optimize edilmiş bir TensorFlow grafiğine dönüştürür.

```python
@tf.function
def train_step(input_l, target_ab):
    # Gradyanları kaydetmek için GradientTape kullan
    with tf.GradientTape() as tape:
        # 1. Modeli çalıştırarak AB kanallarını tahmin et
        predicted_ab = colorization_model(input_l, training=True) # training=True önemli

        # 2. Kaybı hesapla (tahmin edilen ve gerçek AB kanalları arasında)
        loss = mae_loss(target_ab, predicted_ab)

    # 3. Gradyanları hesapla (kayba göre modelin eğitilebilir değişkenleri için)
    gradients = tape.gradient(loss, colorization_model.trainable_variables)

    # 4. Gradyanları uygulayarak modelin ağırlıklarını güncelle
    optimizer.apply_gradients(zip(gradients, colorization_model.trainable_variables))

    # Kaybı takip etmek için geri döndür
    return loss
```

**Açıklama:**
*   `@tf.function`: Python fonksiyonunu optimize edilmiş TensorFlow grafiğine dönüştürür.
*   `train_step(input_l, target_ab)`: Bir batch girdi (L kanalı) ve hedef (AB kanalları) alır.
*   `tf.GradientTape`: Otomatik türev alma için işlemleri kaydeder.
    *   `colorization_model(input_l, training=True)`: Modeli eğitim modunda çalıştırır (örn. Dropout gibi katmanlar aktif olur) ve AB kanallarını tahmin eder.
    *   `mae_loss(target_ab, predicted_ab)`: Tahmin edilen ve gerçek AB kanalları arasındaki Ortalama Mutlak Hatayı hesaplar.
*   `tape.gradient(loss, variables)`: Kayba (`loss`) göre modelin eğitilebilir değişkenlerinin (`colorization_model.trainable_variables`) gradyanlarını hesaplar.
*   `optimizer.apply_gradients(zip(gradients, variables))`: Hesaplanan gradyanları kullanarak optimizasyon algoritması aracılığıyla modelin ağırlıklarını günceller.
*   Fonksiyon, hesaplanan kayıp değerini döndürür, bu değer eğitim ilerlemesini izlemek için kullanılabilir.

---

### Adım 6: Eğitim Döngüsü ve Görselleştirme

Modeli belirtilen sayıda epoch boyunca eğitecek ana döngüyü ve eğitim sırasında modelin performansını görselleştirmek için örnek renklendirilmiş görüntüler üretecek yardımcı fonksiyonu tanımlıyoruz.

```python
# Eğitim parametreleri
EPOCHS = 25 # Toplam eğitim epoch sayısı (daha fazla gerekebilir, örn: 50, 100)
num_examples_to_generate = 4 # Her görselleştirme adımında gösterilecek örnek sayısı

# Görselleştirme için test setinden veya eğitim setinden sabit örnekler alalım
# (processed_L ve processed_AB'nin önceden hesaplandığını varsayıyoruz)
example_indices = np.random.choice(range(len(processed_L)), num_examples_to_generate, replace=False)
example_l_channels = processed_L[example_indices]
example_ab_channels = processed_AB[example_indices]
# Orijinal RGB görüntüleri de saklayalım (denormalize edilmemiş, sadece boyutlandırılmış)
example_original_rgb = []
for idx in example_indices:
    rgb_image_float = tf.cast(all_images[idx], tf.float32) / 255.0
    resized_rgb = transform.resize(rgb_image_float.numpy(), (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    example_original_rgb.append(resized_rgb)
example_original_rgb = np.array(example_original_rgb)


# LAB -> RGB dönüşümü ve görselleştirme fonksiyonu
def generate_and_display_images(model, epoch, input_l, target_ab, original_rgb):
    # Modeli kullanarak AB kanallarını tahmin et (çıkarım modu)
    predicted_ab = model(input_l, training=False)

    # Görüntüleri göstermek için subplot oluştur
    plt.figure(figsize=(10, num_examples_to_generate * 3)) # Boyutu ayarlayabilirsiniz

    for i in range(num_examples_to_generate):
        # Girdi L kanalını al (normalize edilmiş [0, 1])
        l_channel = input_l[i] * 100.0 # Denormalize L -> [0, 100]
        # Tahmin edilen AB kanallarını al (normalize edilmiş [-1, 1])
        pred_ab = predicted_ab[i] * 128.0 # Denormalize AB -> [-128, 127]
        # Gerçek AB kanallarını al (normalize edilmiş [-1, 1])
        true_ab = target_ab[i] * 128.0 # Denormalize AB -> [-128, 127]

        # Tahmin edilen LAB görüntüsünü oluştur
        pred_lab = np.concatenate([l_channel, pred_ab], axis=-1)
        # Gerçek LAB görüntüsünü oluştur (karşılaştırma için)
        true_lab = np.concatenate([l_channel, true_ab], axis=-1)

        # LAB -> RGB dönüşümü
        pred_rgb = color.lab2rgb(pred_lab)
        # Gerçek RGB (LAB'dan dönüştürülmüş - orijinal RGB ile aynı olmalı)
        # true_rgb_from_lab = color.lab2rgb(true_lab) # Bunu da gösterebiliriz
        # Orijinal RGB (veri setinden alınan)
        orig_rgb = original_rgb[i]

        # Girdi (Grayscale L), Tahmin (Renkli), Orijinal (Renkli)
        display_list = [l_channel[:,:,0] / 100.0, pred_rgb, orig_rgb] # L'yi tekrar 0-1 yapıp grayscale göster
        title = ['Girdi (L Kanalı)', 'Tahmin Edilen Renkli', 'Orijinal Renkli']

        for j in range(3):
            plt.subplot(num_examples_to_generate, 3, i * 3 + j + 1)
            plt.title(title[j])
            if j == 0:
                plt.imshow(display_list[j], cmap='gray') # L kanalını grayscale göster
            else:
                # lab2rgb sonucu [0,1] aralığında olmalı ama küçük hatalar olabilir
                plt.imshow(np.clip(display_list[j], 0, 1))
            plt.axis('off')

    plt.suptitle(f"Epoch: {epoch+1}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Başlık için yer bırak
    plt.show()


# Ana Eğitim Fonksiyonu
def train(dataset, epochs):
    history_loss = []

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()

        # Veri seti üzerinde dön
        for batch, (input_l, target_ab) in enumerate(dataset):
            loss = train_step(input_l, target_ab)
            epoch_loss_avg.update_state(loss)

            # İsteğe bağlı: Batch bazında ilerlemeyi gösterme
            # if batch % 100 == 0:
            #     print(f'Epoch {epoch + 1} Batch {batch} Loss {loss:.4f}')

        # Her epoch sonunda ortalama kaybı kaydet ve görselleştir
        current_loss = epoch_loss_avg.result().numpy()
        history_loss.append(current_loss)

        display.clear_output(wait=True) # Önceki çıktıyı temizle (Jupyter için)
        # Görselleştirme için örnekleri kullan
        generate_and_display_images(colorization_model, epoch,
                                    example_l_channels, example_ab_channels, example_original_rgb)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Süre: {time.time() - start_time:.2f} sn')
        print(f'Ortalama Kayıp (MAE): {current_loss:.4f}')
        print('-'*20)

    # Son epoch sonrası tekrar üret
    display.clear_output(wait=True)
    generate_and_display_images(colorization_model, epochs-1, # Epoch 0'dan başladığı için epochs-1
                                example_l_channels, example_ab_channels, example_original_rgb)

    return history_loss

# Eğitimi Başlat
print("Eğitim Başlıyor...")
loss_history = train(train_dataset, EPOCHS)
print("Eğitim Tamamlandı!")

# Kayıp grafiğini çizdirme (Opsiyonel)
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Ortalama Mutlak Hata (MAE)')
plt.title('Eğitim Boyunca Kayıp Değeri')
plt.xlabel('Epoch')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.show()

```

**Açıklama:**
*   `EPOCHS`: Modelin tüm veri seti üzerinden kaç kez geçeceğini belirtir. Renklendirme modelleri genellikle daha fazla epoch gerektirebilir.
*   `num_examples_to_generate`: Her görselleştirme adımında kaç örneğin gösterileceği.
*   Örnek Seçimi: Eğitim veya test setinden rastgele birkaç örnek seçilir ve bunların L, AB kanalları ve orijinal RGB versiyonları saklanır. Bu sabit örnekler, epoch'lar boyunca modelin ilerlemesini karşılaştırmak için kullanılır.
*   `generate_and_display_images`:
    *   Modeli `training=False` modunda çalıştırarak verilen L kanalları için AB kanallarını tahmin eder.
    *   Denormalizasyon: Tahmin edilen ve gerçek AB kanallarını `* 128.0`, girdi L kanalını `* 100.0` ile orijinal LAB değer aralıklarına geri getirir.
    *   `np.concatenate`: L ve AB kanallarını birleştirerek tam LAB görüntülerini oluşturur.
    *   `color.lab2rgb`: Scikit-image fonksiyonu ile LAB görüntülerini RGB'ye dönüştürür.
    *   Görselleştirme: `matplotlib` kullanarak her örnek için yan yana Girdi (L kanalı), Tahmin Edilen Renkli (LAB->RGB) ve Orijinal Renkli (RGB) görüntüleri gösterir. L kanalı `cmap='gray'` ile siyah beyaz gösterilir. `np.clip` ile RGB değerlerinin [0, 1] aralığında kalması sağlanır.
*   `train`: Ana eğitim döngüsü.
    *   Her epoch'ta veri seti üzerinde döner ve her batch için `train_step` fonksiyonunu çağırır.
    *   `tf.keras.metrics.Mean`: Epoch boyunca batch kayıplarının ortalamasını kolayca hesaplamak için kullanılır.
    *   `display.clear_output(wait=True)`: Jupyter Notebook'ta çıktıyı temizler.
    *   `generate_and_display_images` fonksiyonunu çağırarak modelin o anki performansını örnekler üzerinde gösterir.
    *   Epoch süresini ve ortalama kaybı yazdırır.
*   Eğitim bittikten sonra, isteğe bağlı olarak kayıp değerinin epoch'lara göre değişimini gösteren bir grafik çizdirilir.

---

### Adım 7: Yeni Görüntüleri Renklendirme

Eğitim tamamlandıktan sonra, `colorization_model`'i kullanarak daha önce görmediği siyah beyaz görüntüleri veya renkli görüntülerin siyah beyaz versiyonlarını renklendirebiliriz.

Bunun için şu adımları izleyen bir fonksiyon yazacağız:
1.  Görüntüyü yükle.
2.  Görüntüyü modelin eğitildiği boyuta (`IMG_SIZE`) getir.
3.  Görüntüyü LAB renk uzayına çevir.
4.  L kanalını ayır ve normalize et (modelin girdisi için).
5.  Modeli kullanarak normalize edilmiş L kanalından normalize edilmiş AB kanallarını tahmin et.
6.  Girdi L kanalını ve tahmin edilen AB kanallarını denormalize et.
7.  Denormalize edilmiş L ve AB kanallarını birleştirerek tam LAB görüntüsünü oluştur.
8.  LAB görüntüsünü tekrar RGB'ye çevir.
9.  Orijinal siyah beyaz girdiyi ve renklendirilmiş çıktıyı göster.

```python
def colorize_image(image_path, model, img_size):
    """Verilen yoldaki görüntüyü yükler, renklendirir ve gösterir."""
    try:
        # Görüntüyü yükle (skimage.io.imread renkli (RGB) veya grayscale okuyabilir)
        img_rgb = io.imread(image_path)

        # Görüntü grayscale ise 3 kanala çıkar (RGB gibi)
        if img_rgb.ndim == 2:
            img_rgb = color.gray2rgb(img_rgb)
        # Görüntü RGBA ise alpha kanalını at
        elif img_rgb.shape[2] == 4:
            img_rgb = img_rgb[:, :, :3]

        # Görüntüyü float [0, 1] aralığına getir ve boyutlandır
        img_rgb_float = tf.cast(img_rgb, tf.float32) / 255.0
        resized_rgb = transform.resize(img_rgb_float.numpy(), (img_size, img_size), anti_aliasing=True)

        # RGB -> LAB
        img_lab = color.rgb2lab(resized_rgb)

        # L kanalını ayır ve normalize et ([0, 1])
        l_channel = img_lab[:, :, 0]
        l_channel_normalized = l_channel / 100.0

        # Model girdisi için batch ve kanal boyutu ekle: (H, W) -> (1, H, W, 1)
        input_l = tf.expand_dims(tf.expand_dims(l_channel_normalized, axis=-1), axis=0)

        # Modeli kullanarak AB kanallarını tahmin et (çıkarım modu)
        predicted_ab_normalized = model(input_l, training=False)[0] # Batch boyutunu kaldır

        # Tahmin edilen AB kanallarını denormalize et ([-1, 1] -> [-128, 127])
        predicted_ab = predicted_ab_normalized * 128.0

        # Girdi L kanalını denormalize et ([0, 1] -> [0, 100])
        # (Zaten img_lab içinde denormalize hali var, onu kullanalım)
        l_channel_denorm = np.expand_dims(l_channel, axis=-1) # Kanal boyutu ekle

        # Denormalize L ve tahmin edilen AB'yi birleştir
        predicted_lab = np.concatenate([l_channel_denorm, predicted_ab.numpy()], axis=-1)

        # LAB -> RGB dönüşümü
        predicted_rgb = color.lab2rgb(predicted_lab)

        # Sonuçları göster
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Girdi (Siyah Beyaz)")
        # Orijinal L kanalını grayscale olarak göster
        plt.imshow(l_channel / 100.0, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Model Çıktısı (Renklendirilmiş)")
        # Tahmin edilen RGB'yi göster (clip ile [0, 1] aralığında kalmasını sağla)
        plt.imshow(np.clip(predicted_rgb, 0, 1))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Hata: '{image_path}' bulunamadı.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")


# Kullanım Örneği:
# Test setinden bir görüntü alalım veya kendi görüntünüzün yolunu verin
# test_image_index = 0 # Test setinden bir örnek seç
# test_rgb_image = test_images[test_image_index]
# io.imsave("temp_test_image.png", test_rgb_image) # Geçici olarak kaydet
# image_to_colorize_path = "temp_test_image.png"

# Veya kendi dosya yolunuzu belirtin:
image_to_colorize_path = 'Data/test.jpg' # Bu dosyanın var olduğundan emin olun!

# Eğitilmiş modeli kullanarak renklendirme yap
colorize_image(image_to_colorize_path, colorization_model, IMG_SIZE)

# Geçici dosyayı sil (eğer oluşturulduysa)
# if os.path.exists("temp_test_image.png"):
#     os.remove("temp_test_image.png")

```

**Açıklama:**
*   `colorize_image`: Görüntü yolu, eğitilmiş model ve hedef boyutu alır.
    *   `io.imread`: Görüntüyü yükler.
    *   Görüntü Kontrolü: Görüntünün grayscale veya RGBA olup olmadığını kontrol eder ve 3 kanallı RGB formatına getirir.
    *   Ön İşleme: Görüntüyü [0, 1] aralığına getirir, boyutlandırır ve LAB'a dönüştürür.
    *   L Kanalı Hazırlama: L kanalını ayırır, normalize eder ve modelin beklediği `(1, H, W, 1)` şekline getirir.
    *   Tahmin: Modeli `training=False` modunda çalıştırarak normalize edilmiş AB kanallarını tahmin eder. Batch boyutu `[0]` ile kaldırılır.
    *   Denormalizasyon: Tahmin edilen AB kanalları `* 128.0` ile denormalize edilir. Orijinal L kanalı (zaten [0, 100] aralığında) kullanılır.
    *   Birleştirme ve Dönüşüm: Denormalize L ve tahmin edilen AB kanalları birleştirilerek LAB görüntüsü oluşturulur ve `color.lab2rgb` ile RGB'ye dönüştürülür.
    *   Görselleştirme: Orijinal L kanalı (siyah beyaz) ve modelin ürettiği renklendirilmiş RGB görüntüsü yan yana gösterilir.
*   Kullanım Örneği: Renklendirilecek görüntünün yolu (`image_to_colorize_path`) belirtilir ve `colorize_image` fonksiyonu çağrılır. Örnekte `Data/test.jpg` kullanılmıştır, bu dosyanın projenizde mevcut olması veya kendi dosya yolunuzu vermeniz gerekir.

Artık bu kod bloklarını sırasıyla çalıştırarak kendi siyah beyaz görüntü renklendirme modelinizi eğitebilir ve test edebilirsiniz! Unutmayın ki CIFAR-10 gibi küçük bir veri setiyle eğitilen modelin performansı sınırlı olacaktır. Daha iyi sonuçlar için daha büyük ve çeşitli veri setleri üzerinde daha uzun süre eğitmeniz gerekebilir.
