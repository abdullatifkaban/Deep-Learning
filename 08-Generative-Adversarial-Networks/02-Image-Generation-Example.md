## 8.2 Örnek Uygulama: Elma Görüntüsü Üretme

Bu bölümde, GAN kullanarak `downloads/elma/` klasöründeki elma görüntülerine benzer yeni elma görüntüleri üretmeye çalışan bir örnek yapacağız. Adımları takip ederek ve kod bloklarını çalıştırarak kendi GAN modelinizi eğitebilirsiniz.

**Genel Akış:**

1.  Gerekli kütüphaneleri yükleyeceğiz.
2.  `downloads/elma/` klasöründeki elma görüntülerini yükleyip hazırlayacağız.
3.  Üretici (Generator) modelini tanımlayacağız.
4.  Ayırt Edici (Discriminator) modelini tanımlayacağız.
5.  Modeller için kayıp fonksiyonlarını ve optimizasyon algoritmalarını belirleyeceğiz.
6.  Eğitim adımlarını tanımlayacağız.
7.  Modeli eğiteceğiz ve üretilen görüntüleri gözlemleyeceğiz.

---

### Adım 1: Kütüphanelerin Yüklenmesi

Gerekli olan TensorFlow, Keras, NumPy, Matplotlib ve OS kütüphanelerini projemize dahil ediyoruz.

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from IPython import display # Görüntüleri notebook'ta göstermek için
```

**Açıklama:**
*   `tensorflow` ve `keras.layers`: Derin öğrenme modellerini oluşturmak ve eğitmek için temel kütüphaneler.
*   `numpy`: Sayısal işlemler için.
*   `matplotlib.pyplot`: Görüntüleri görselleştirmek için.
*   `os`: Dosya ve dizin işlemleri için (veri setinin yolunu belirtmek vb.).
*   `time`: Eğitim süresini ölçmek için.
*   `IPython.display`: Jupyter Notebook gibi ortamlarda eğitim sırasında üretilen görüntüleri temiz bir şekilde göstermek için.

---

### Adım 2: Veri Setinin Yüklenmesi ve Hazırlanması

> Elma resimlerini [buradaki dosyayı](../Data/elma.zip) dosyadan elde edebilirsiniz. Resimleri `elma` isimli klasöre atıp devam ediniz.

`elma/` klasöründeki elma görüntülerini yükleyip, modelin işleyebileceği formata getireceğiz. Görüntüleri belirli bir boyuta (örneğin 64x64) yeniden boyutlandıracak ve piksel değerlerini -1 ile 1 arasına normalize edeceğiz.

```python
# Veri seti yolu ve parametreler
DATASET_PATH = 'elma/'
IMG_WIDTH = 64
IMG_HEIGHT = 64
BUFFER_SIZE = 500 # Veri karıştırma tampon boyutu (veri setindeki örnek sayısına yakın olabilir)
BATCH_SIZE = 64   # Her eğitim adımında kullanılacak örnek sayısı

# Görüntüleri yükleme ve ön işleme fonksiyonu
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3) # Görüntüyü JPEG olarak oku (3 kanal - RGB)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH]) # Yeniden boyutlandır
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5 # Normalize to [-1, 1]
    return image

# Veri setini oluşturma
image_paths = [os.path.join(DATASET_PATH, fname) for fname in os.listdir(DATASET_PATH) if fname.endswith('.jpg')]
print(f"Toplam {len(image_paths)} elma görüntüsü bulundu.")

# tf.data.Dataset oluşturma
path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Veri setini karıştırma ve batch'lere ayırma
train_dataset = image_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print("Veri seti hazırlandı. Örnek bir batch şekli:", next(iter(train_dataset)).shape)
```

**Açıklama:**
*   `DATASET_PATH`: Elma görüntülerinin bulunduğu klasörün yolu.
*   `IMG_WIDTH`, `IMG_HEIGHT`: Tüm görüntülerin getirileceği standart boyut. GAN'lar genellikle sabit boyutlu girdi/çıktı ile çalışır. 64x64 başlangıç için iyi bir boyuttur.
*   `BUFFER_SIZE`, `BATCH_SIZE`: `tf.data.Dataset` için performans ve eğitim dinamiklerini etkileyen parametreler.
*   `load_and_preprocess_image`: Tek bir görüntünün yolunu alıp, okuyan, yeniden boyutlandıran ve normalize eden fonksiyon.
    *   `tf.io.read_file`: Görüntü dosyasını okur.
    *   `tf.image.decode_jpeg`: JPEG formatındaki baytları tensor'a çevirir (`channels=3` RGB olduğunu belirtir).
    *   `tf.image.resize`: Görüntüyü hedef boyuta getirir.
    *   Normalizasyon: Piksel değerlerini (0-255 aralığında) -1 ile 1 aralığına getirir. Bu, genellikle generator'ın son katmanında `tanh` aktivasyonu kullanıldığında tercih edilir.
*   `os.listdir` ve `os.path.join`: Klasördeki tüm `.jpg` dosyalarının tam yollarını listeler.
*   `tf.data.Dataset.from_tensor_slices`: Dosya yollarından bir dataset oluşturur.
*   `.map`: Her dosya yoluna `load_and_preprocess_image` fonksiyonunu uygular. `num_parallel_calls=tf.data.AUTOTUNE` ile işlemi hızlandırır.
*   `.shuffle`: Veri setini karıştırarak modelin örneklerin sırasına bağımlı kalmasını engeller.
*   `.batch`: Veri setini belirtilen boyutlarda gruplara (batch) ayırır.

---

### Adım 3: Üretici (Generator) Modelinin Oluşturulması

Rastgele gürültü vektörünü (latent space) girdi olarak alıp, hedef boyutta (64x64x3) bir elma görüntüsü üretmeye çalışacak modeli tanımlıyoruz. Genellikle `Dense` katman ile başlar ve `Conv2DTranspose` (bazen Deconvolution olarak da adlandırılır) katmanları ile görüntüyü büyütür.

```python
LATENT_DIM = 100 # Rastgele gürültü vektörünün boyutu

def make_generator_model(img_height, img_width):
    model = tf.keras.Sequential()
    # Temel Dense katman ve yeniden şekillendirme
    # 4x4x1024 boyutunda bir başlangıç tensörü oluşturuyoruz
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)

    # Conv2DTranspose katmanları ile boyutu artırma (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64)
    # 8x8
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 16x16
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 32x32
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 64x64 - Son katman
    # Hedef boyut (IMG_HEIGHT, IMG_WIDTH) ve 3 renk kanalı (RGB)
    # Aktivasyon olarak 'tanh' kullanıyoruz çünkü piksel değerlerini -1 ile 1 arasına normalize ettik.
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, img_height, img_width, 3)

    return model

generator = make_generator_model(IMG_HEIGHT, IMG_WIDTH)
generator.summary() # Model özetini göster

# Test: Rastgele gürültü ile bir görüntü üretelim (eğitimden önce)
noise = tf.random.normal([1, LATENT_DIM])
generated_image = generator(noise, training=False)
plt.imshow((generated_image[0] + 1) / 2) # Görüntüyü 0-1 aralığına getirip göster
plt.title("Eğitimsiz Üretici Çıktısı")
plt.axis('off')
plt.show()
```

**Açıklama:**
*   `LATENT_DIM`: Üreticinin başlangıç noktası olan rastgele vektörün boyutu. 100 yaygın bir değerdir.
*   `make_generator_model`: Modeli oluşturan fonksiyon.
    *   `layers.Dense`: Gürültü vektörünü alır ve daha büyük bir boyuta genişletir.
    *   `layers.BatchNormalization`: Her katmanın çıktısını normalize ederek eğitimi stabilize eder ve hızlandırır.
    *   `layers.LeakyReLU`: Sıfır olmayan negatif eğime sahip bir aktivasyon fonksiyonu, ReLU'ya göre "ölü nöron" sorununu azaltmaya yardımcı olabilir.
    *   `layers.Reshape`: Dense katmanın çıktısını, konvolüsyonel katmanların işleyebileceği 3D bir formata (yükseklik, genişlik, kanal) dönüştürür.
    *   `layers.Conv2DTranspose`: Transpoze konvolüsyon (veya dekonvolüsyon). Görüntünün uzamsal boyutlarını (yükseklik ve genişlik) artırmak için kullanılır. `strides=(2, 2)` genellikle boyutu iki katına çıkarır.
    *   Son `Conv2DTranspose` katmanı: Çıktı kanal sayısını 3 (RGB için) yapar ve `tanh` aktivasyonunu kullanır. `tanh` fonksiyonu çıktıları -1 ile 1 aralığında verir, bu da normalize ettiğimiz gerçek görüntülerle uyumludur.
*   `generator.summary()`: Modelin katmanlarını ve parametre sayısını gösterir.
*   Test kodu: Eğitim başlamadan önce, rastgele başlatılmış ağırlıklara sahip üreticinin ne tür bir "gürültü" ürettiğini görmek için küçük bir test yaparız. Çıktıyı görselleştirmeden önce `(image + 1) / 2` ile 0-1 aralığına geri getiririz.

---

### Adım 4: Ayırt Edici (Discriminator) Modelinin Oluşturulması

Girdi olarak bir görüntü (gerçek veya sahte, 64x64x3 boyutunda) alıp, bu görüntünün gerçek olma olasılığını tahmin eden (bir skor üreten) bir model tanımlıyoruz. Bu genellikle standart bir CNN (Convolutional Neural Network) sınıflandırma modeline benzer.

```python
def make_discriminator_model(img_height, img_width):
    model = tf.keras.Sequential()
    input_shape = [img_height, img_width, 3]

    # Conv2D katmanları ile özellik çıkarma (64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4)
    # 32x32
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) # Dropout, overfitting'i azaltmaya yardımcı olur

    # 16x16
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # 8x8
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # 4x4
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Sınıflandırma katmanı
    model.add(layers.Flatten()) # Özellik haritalarını düzleştirir
    model.add(layers.Dense(1)) # Tek bir çıktı nöronu (gerçek/sahte skoru - logit)

    return model

discriminator = make_discriminator_model(IMG_HEIGHT, IMG_WIDTH)
discriminator.summary()

# Test: Eğitimsiz Ayırt Edici'nin ürettiği skoru görelim
decision = discriminator(generated_image, training=False) # Önceki adımda üretilen sahte görüntü
print("Eğitimsiz Ayırt Edici Skoru:", decision.numpy())
```

**Açıklama:**
*   `make_discriminator_model`: Ayırt edici modeli oluşturan fonksiyon.
    *   `input_shape`: Modelin beklediği girdi görüntüsünün boyutu (yükseklik, genişlik, kanal sayısı).
    *   `layers.Conv2D`: Standart konvolüsyon katmanı. Görüntüdeki özellikleri (kenarlar, dokular vb.) çıkarmak için kullanılır. `strides=(2, 2)` genellikle görüntünün boyutunu yarıya indirir.
    *   `layers.LeakyReLU`: Aktivasyon fonksiyonu.
    *   `layers.Dropout`: Eğitim sırasında rastgele bazı nöronları devre dışı bırakarak modelin aşırı öğrenmesini (overfitting) engellemeye yardımcı olan bir regülarizasyon tekniğidir.
    *   `layers.Flatten`: Son konvolüsyon katmanının çıktısı olan çok boyutlu tensörü tek boyutlu bir vektöre dönüştürür.
    *   `layers.Dense(1)`: Tek bir çıktı nöronu. Bu nöron, girdinin gerçek olma olasılığını temsil eden bir skor (logit) üretir. Aktivasyon fonksiyonu (sigmoid gibi) burada kullanılmaz çünkü kayıp fonksiyonu (`BinaryCrossentropy(from_logits=True)`) bu skoru doğrudan kullanacak şekilde tasarlanmıştır. Bu, sayısal olarak daha kararlı olabilir.
*   Test kodu: Eğitimsiz ayırt edicinin, eğitimsiz üreticinin ürettiği sahte görüntüye ne skor verdiğini görürüz.

---

### Adım 5: Kayıp Fonksiyonları ve Optimizasyon

Hem Üretici hem de Ayırt Edici için kayıp (loss) fonksiyonlarını ve bu kayıpları minimize edecek optimizasyon algoritmalarını tanımlamamız gerekiyor.

```python
# Kayıp fonksiyonu: Binary Cross-Entropy (from_logits=True çünkü modellerimiz skor üretiyor)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Ayırt Edici Kaybı
def discriminator_loss(real_output, fake_output):
    # Gerçek görüntülerin skorları 1'e (gerçek etiketi) ne kadar yakın?
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Sahte görüntülerin skorları 0'a (sahte etiketi) ne kadar yakın?
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # Toplam kayıp, bu iki kaybın toplamıdır.
    total_loss = real_loss + fake_loss
    return total_loss

# Üretici Kaybı
def generator_loss(fake_output):
    # Üreticinin amacı, Ayırt Edici'yi kandırmak.
    # Yani, ürettiği sahte görüntülerin skorlarının 1'e (gerçek etiketi) ne kadar yakın olmasını istiyoruz.
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizasyon Algoritmaları (Adam genellikle GAN'lar için iyi çalışır)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

# Not: beta_1=0.5 değeri DCGAN makalesinde önerilen bir değerdir ve bazen eğitimi stabilize edebilir.
```

**Açıklama:**
*   `tf.keras.losses.BinaryCrossentropy(from_logits=True)`: İkili sınıflandırma problemleri için standart kayıp fonksiyonu. `from_logits=True` ayarı, modelin son katmanında sigmoid aktivasyonu olmadığını ve doğrudan skor (logit) ürettiğini belirtir.
*   `discriminator_loss`: Ayırt edicinin ne kadar iyi çalıştığını ölçer. İki bölümden oluşur:
    *   Gerçek görüntülere verdiği skorların hedef olan 1'e ne kadar uzak olduğu (`real_loss`).
    *   Sahte görüntülere verdiği skorların hedef olan 0'a ne kadar uzak olduğu (`fake_loss`).
    *   Ayırt edici bu toplam kaybı minimize etmeye çalışır.
*   `generator_loss`: Üreticinin ne kadar iyi çalıştığını (yani Ayırt Edici'yi ne kadar iyi kandırdığını) ölçer.
    *   Üretici, ürettiği sahte görüntüler için Ayırt Edici'nin verdiği skorların hedef olan 1'e (gerçek etiketi) yaklaşmasını ister. Bu kaybı minimize etmeye çalışır.
*   `tf.keras.optimizers.Adam`: Gradyan inişi tabanlı popüler bir optimizasyon algoritması. `learning_rate` (öğrenme oranı) ve `beta_1` gibi hiperparametreler eğitim performansını etkileyebilir. Düşük öğrenme oranı (örn. `1e-4`) ve `beta_1=0.5` GAN eğitiminde sıkça kullanılır.

---

### Adım 6: Eğitim Adımının Tanımlanması

Tek bir eğitim adımında hem Ayırt Edici'nin hem de Üretici'nin nasıl güncelleneceğini tanımlayan fonksiyonu yazıyoruz. `@tf.function` dekoratörü, bu Python fonksiyonunu optimize edilmiş bir TensorFlow grafiğine dönüştürerek eğitimi hızlandırır.

```python
@tf.function
def train_step(images):
    # 1. Üretici için rastgele gürültü oluştur
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    # 2. Gradyanları kaydetmek için GradientTape kullan
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 2a. Üretici ile sahte görüntüler üret
        generated_images = generator(noise, training=True)

        # 2b. Ayırt Edici ile gerçek ve sahte görüntülerin skorlarını al
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # 2c. Kayıpları hesapla
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # 3. Gradyanları hesapla
    # Üretici gradyanları (gen_loss'a göre)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Ayırt Edici gradyanları (disc_loss'a göre)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 4. Gradyanları uygulayarak modellerin ağırlıklarını güncelle
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Kayıpları takip etmek için geri döndür (opsiyonel)
    return gen_loss, disc_loss
```

**Açıklama:**
*   `@tf.function`: TensorFlow'un AutoGraph özelliği. Python kodunu yüksek performanslı TensorFlow grafiğine derler.
*   `tf.random.normal`: Belirtilen boyutta (`BATCH_SIZE`, `LATENT_DIM`) normal dağılımdan rastgele gürültü vektörleri oluşturur.
*   `tf.GradientTape`: Otomatik türev alma (automatic differentiation) için kullanılır. Bu blok içindeki işlemlerin gradyanları kaydedilir. Üretici ve Ayırt Edici için ayrı `tape`'ler kullanırız.
    *   `generator(noise, training=True)`: Üreticiyi eğitim modunda çalıştırır (örn. BatchNormalization ve Dropout katmanları aktif olur).
    *   `discriminator(..., training=True)`: Ayırt ediciyi eğitim modunda çalıştırır.
    *   Kayıp fonksiyonları çağrılarak kayıplar hesaplanır.
*   `tape.gradient(loss, variables)`: Belirtilen kayba (`loss`) göre, belirtilen değişkenlerin (`variables`) gradyanlarını hesaplar.
*   `optimizer.apply_gradients(zip(gradients, variables))`: Hesaplanan gradyanları kullanarak optimizasyon algoritması aracılığıyla modelin ağırlıklarını günceller.

---

### Adım 7: Eğitim Döngüsü ve Görüntü Üretme

Modeli belirtilen sayıda epoch boyunca eğitecek ana döngüyü ve eğitim sırasında belirli aralıklarla örnek görüntüler üretecek yardımcı fonksiyonu tanımlıyoruz.

```python
# Eğitim parametreleri
EPOCHS = 100 # Toplam eğitim epoch sayısı (daha fazla gerekebilir)
num_examples_to_generate = 16 # Her kaydetme adımında üretilecek örnek sayısı

# Görüntü üretme ve kaydetme için sabit gürültü vektörü (ilerlemeyi görmek için)
seed = tf.random.normal([num_examples_to_generate, LATENT_DIM])

# Görüntüleri üreten ve gösteren/kaydeden fonksiyon
def generate_and_save_images(model, epoch, test_input):
    # `training=False` önemli, model çıkarım modunda çalışır (örn. BatchNormalization farklı davranır)
    predictions = model(test_input, training=False)
    # Görüntüleri 4x4 grid'de gösterme
    fig = plt.figure(figsize=(8, 8)) # Boyutu ayarlayabilirsiniz

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Piksel değerlerini 0-1 aralığına getir
        img_to_show = (predictions[i, :, :, :] + 1) / 2.0
        plt.imshow(img_to_show)
        plt.axis('off')

    # İsteğe bağlı: Görüntüleri dosyaya kaydetme
    # output_dir = 'gan_generated_images'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # plt.savefig(os.path.join(output_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))

    plt.suptitle(f"Epoch: {epoch}", fontsize=16)
    plt.show()


# Ana Eğitim Fonksiyonu
def train(dataset, epochs):
    history_gen_loss = []
    history_disc_loss = []

    for epoch in range(epochs):
        start_time = time.time()
        epoch_gen_loss = []
        epoch_disc_loss = []

        # Veri seti üzerinde dön
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_gen_loss.append(gen_loss.numpy())
            epoch_disc_loss.append(disc_loss.numpy())

        # Epoch sonunda ortalama kayıpları hesapla
        avg_gen_loss = np.mean(epoch_gen_loss)
        avg_disc_loss = np.mean(epoch_disc_loss)
        history_gen_loss.append(avg_gen_loss)
        history_disc_loss.append(avg_disc_loss)

        # Her epoch sonunda veya belirli aralıklarla örnek üret ve göster
        display.clear_output(wait=True) # Önceki çıktıyı temizle (Jupyter için)
        generate_and_save_images(generator, epoch + 1, seed)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Süre: {time.time() - start_time:.2f} sn')
        print(f'Generator Loss: {avg_gen_loss:.4f}')
        print(f'Discriminator Loss: {avg_disc_loss:.4f}')
        print('-'*20)

    # Son epoch sonrası tekrar üret
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

    return history_gen_loss, history_disc_loss

# Eğitimi Başlat
print("Eğitim Başlıyor...")
gen_loss_history, disc_loss_history = train(train_dataset, EPOCHS)
print("Eğitim Tamamlandı!")

# Kayıp grafiğini çizdirme (Opsiyonel)
plt.figure(figsize=(10, 5))
plt.plot(gen_loss_history, label='Generator Loss')
plt.plot(disc_loss_history, label='Discriminator Loss')
plt.title('Eğitim Boyunca Kayıp Değerleri')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Açıklama:**
*   `EPOCHS`: Modelin tüm veri seti üzerinden kaç kez geçeceğini belirtir. GAN eğitimleri genellikle uzun sürer (yüzlerce veya binlerce epoch). Başlangıç için 100 iyi bir değer olabilir, ancak daha iyi sonuçlar için artırmanız gerekebilir.
*   `seed`: Her zaman aynı başlangıç gürültü vektörünü kullanarak görüntü üretiriz. Bu, farklı epoch'lardaki çıktıları karşılaştırarak modelin ilerlemesini tutarlı bir şekilde görmemizi sağlar.
*   `generate_and_save_images`:
    *   `model(test_input, training=False)`: Üreticiyi çıkarım modunda çalıştırır.
    *   Üretilen görüntüleri `matplotlib` kullanarak 4x4'lük bir grid üzerinde gösterir. Piksel değerleri `(img + 1) / 2` ile tekrar 0-1 aralığına getirilir.
    *   Yorum satırı halindeki `plt.savefig` kısmı, isterseniz üretilen görüntüleri her epoch sonunda bir dosyaya kaydetmenizi sağlar.
*   `train`: Ana eğitim döngüsü.
    *   Her epoch'ta veri seti üzerinde döner ve her batch için `train_step` fonksiyonunu çağırır.
    *   Epoch içindeki kayıpların ortalamasını alır ve kaydeder.
    *   `display.clear_output(wait=True)`: Jupyter Notebook gibi ortamlarda, her epoch'ta önceki grafiği temizleyip yenisini göstererek daha düzenli bir çıktı sağlar.
    *   `generate_and_save_images` fonksiyonunu çağırarak o anki üretici modelinin ürettiği örnekleri gösterir.
    *   Epoch süresini ve ortalama kayıpları yazdırır.
*   Eğitim bittikten sonra, isteğe bağlı olarak Üretici ve Ayırt Edici kayıplarının epoch'lara göre değişimini gösteren bir grafik çizdirilir. Bu grafik, eğitimin nasıl gittiği (örneğin, kayıpların dengede olup olmadığı, çökme olup olmadığı) hakkında fikir verir.

---

Artık bu kod bloklarını sırasıyla çalıştırarak kendi elma görüntülerinizi üreten GAN modelinizi eğitebilirsiniz! Unutmayın, iyi sonuçlar almak için `EPOCHS` sayısını artırmanız ve belki de model mimarisi veya hiperparametrelerle (öğrenme oranı, batch boyutu vb.) oynamanız gerekebilir.

---

### Adım 8: Eğitilmiş Modelden Yeni Görüntü Üretme

Eğitim tamamlandıktan sonra, eğitilmiş `generator` modelini kullanarak istediğiniz zaman yeni elma görüntüleri üretebilirsiniz. Bunun için tek yapmanız gereken rastgele bir gürültü vektörü oluşturup modele vermektir.

```python
# Yeni bir rastgele gürültü vektörü oluştur
new_noise = tf.random.normal([1, LATENT_DIM])

# Modeli kullanarak görüntüyü üret (training=False unutmayın!)
new_generated_image = generator(new_noise, training=False)

# Görüntüyü görselleştirmek için 0-1 aralığına getir
img_to_display = (new_generated_image[0] + 1) / 2.0

# Görüntüyü göster
plt.imshow(img_to_display)
plt.title("Eğitilmiş Model Tarafından Üretilen Yeni Elma")
plt.axis('off')
plt.show()
```

**Açıklama:**
*   `tf.random.normal([1, LATENT_DIM])`: Tek bir örnek (`[1, ...]`) için rastgele gürültü vektörü oluşturur.
*   `generator(new_noise, training=False)`: Eğitilmiş üreticiyi kullanarak gürültüden görüntü üretir. `training=False` modunda çalıştırmak önemlidir.
*   `(new_generated_image[0] + 1) / 2.0`: Üretilen görüntünün piksel değerlerini [-1, 1] aralığından [0, 1] aralığına dönüştürür, böylece `matplotlib` ile doğru şekilde görüntülenebilir.
*   `plt.imshow(...)`: Görüntüyü ekranda gösterir.
