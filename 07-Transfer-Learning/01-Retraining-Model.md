# Modelin Yeniden Eğitimi

Bu eğitimde, TensorFlow ve Keras kullanarak sıfırdan bir Evrişimli Sinir Ağı (Convolutional Neural Network - CNN) ile kedi ve köpek resimlerini sınıflandıran bir model eğiteceğiz. Ardından, bu eğitilmiş modeli kullanarak yeni bir resmi tahmin edeceğiz ve eğer model yanlış tahmin yaparsa, o resmi kullanarak modeli nasıl yeniden eğitebileceğimizi (fine-tuning/transfer learning konseptine benzer bir yaklaşım) göreceğiz.

## Bölüm 1: Sıfırdan Resim Sınıflandırma Modeli Eğitimi

Bu bölümde, bir resim sınıflandırma modelinin nasıl oluşturulduğunu ve eğitildiğini anlayacağız.

### 1. Gerekli Kütüphanelerin İçe Aktarılması

İlk olarak, projemizde kullanacağımız kütüphaneleri içe aktarıyoruz.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib
matplotlib.use('Agg') # Grafik çizimi için arka planı ayarla (Sunucusuz ortamlar için önemli)
import matplotlib.pyplot as plt
```

*   `tensorflow` ve `keras`: Derin öğrenme modeli oluşturmak, eğitmek ve değerlendirmek için ana kütüphaneler.
*   `layers`: Modelimizin katmanlarını (Conv2D, MaxPooling2D, Dense vb.) tanımlamak için kullanılır.
*   `os`: İşletim sistemiyle ilgili işlemler (dosya yolları vb.) için kullanılır.
*   `matplotlib`: Eğitim sonuçlarını görselleştirmek için kullanılır. `matplotlib.use('Agg')` komutu, grafiklerin bir grafik arayüzü olmayan ortamlarda da (örneğin sunucularda) sorunsuz oluşturulup dosyaya kaydedilmesini sağlar.

### 2. Parametrelerin Tanımlanması

Model eğitimi ve veri işleme için gerekli temel parametreleri belirliyoruz.

```python
# --- Parametreler ---
img_height = 180
img_width = 180
batch_size = 32
data_dir = 'dataset'
model_save_path = 'cat_dog_classifier.keras' # Modelin kaydedileceği dosya adı (.keras formatı)
epochs = 30 # Eğitim döngüsü sayısı
```

*   `img_height`, `img_width`: Modele verilecek resimlerin boyutları. Tüm resimler bu boyutlara getirilecektir.
*   `batch_size`: Modelin her adımda kaç resmi işleyeceği.
*   `data_dir`: Resim veri setinin bulunduğu ana klasör ('dataset' klasörü altında 'cat' ve 'dog' alt klasörleri olmalı).
*   `model_save_path`: Eğitilen modelin kaydedileceği dosya yolu ve adı. `.keras` formatı tavsiye edilir.
*   `epochs`: Modelin tüm veri seti üzerinden kaç kez eğitileceği.

### 3. Veri Yükleme ve Hazırlama

Keras'ın `image_dataset_from_directory` yardımcı fonksiyonunu kullanarak diskteki resimleri yükleyip eğitim ve doğrulama (validation) setlerine ayırıyoruz.

```python
# --- Veri Yükleme ve Hazırlama ---
print("Veri yükleniyor...")
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # Verinin %20'sini doğrulama için ayır
  subset="training",      # Bu kısmı eğitim için kullan
  seed=123,             # Tekrarlanabilirlik için rastgelelik tohumu
  image_size=(img_height, img_width), # Resimleri belirtilen boyuta getir
  batch_size=batch_size)  # Veriyi batch'lere ayır

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # Aynı ayırma oranını kullan
  subset="validation",    # Bu kısmı doğrulama için kullan
  seed=123,             # Aynı tohumu kullan (aynı ayırma için)
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names # Sınıf isimlerini al (klasör isimlerinden)
print("Sınıflar:", class_names)
num_classes = len(class_names)
```

*   `image_dataset_from_directory`: Belirtilen klasör yapısına göre (her alt klasör bir sınıf) resimleri yükler ve etiketler.
*   `validation_split` ve `subset`: Veri setini eğitim ve doğrulama olarak ikiye ayırmak için kullanılır.
*   `seed`: Veri ayırma işleminin her çalıştırmada aynı olmasını sağlar.
*   `class_names`: Veri kümesindeki sınıfların isimlerini (örneğin `['cat', 'dog']`) içerir.

### 4. Veri Artırma (Data Augmentation)

Modelin genelleme yeteneğini artırmak ve ezberlemesini (overfitting) azaltmak için mevcut resimlere rastgele dönüşümler uygulayan bir katman oluşturuyoruz.

```python
# Veri artırma (Data Augmentation) katmanı
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)), # Yatay çevirme
    layers.RandomRotation(0.1), # Rastgele döndürme
    layers.RandomZoom(0.1),    # Rastgele yakınlaştırma
  ]
)
```

*   Bu katman, eğitim sırasında her resme rastgele olarak yatay çevirme, döndürme ve yakınlaştırma uygular. Bu sayede model, aynı nesnenin farklı görünümlerini öğrenir.
*   `input_shape`: Bu katmanın modelin ilk katmanı olacağını ve beklenen girdi şeklini belirtir.

### 5. Veri Kümesi Performans Optimizasyonu

Eğitim sırasında veri yükleme darboğazlarını önlemek için veri kümelerini önbelleğe alıyoruz (`cache`) ve bir sonraki adımlar için verileri önceden hazırlıyoruz (`prefetch`).

```python
# Performans için veri kümelerini yapılandırma
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

*   `cache()`: Veriyi ilk epoch'tan sonra bellekte veya diskte tutarak sonraki epoch'larda daha hızlı erişim sağlar.
*   `shuffle(1000)`: Eğitim verisini karıştırarak modelin verinin sırasına bağımlı kalmasını engeller.
*   `prefetch(buffer_size=AUTOTUNE)`: Model mevcut batch'i işlerken bir sonraki batch'i arka planda hazırlar, böylece GPU/CPU bekleme süresini azaltır.

### 6. Model Oluşturma (CNN Mimarisi)

Sıfırdan bir Evrişimli Sinir Ağı (CNN) modeli tanımlıyoruz.

```python
# --- Model Oluşturma (Sıfırdan CNN) ---
print("Model oluşturuluyor...")
model = keras.Sequential([
  data_augmentation, # İlk katman olarak veri artırmayı ekle
  layers.Rescaling(1./255), # Piksel değerlerini [0,1] aralığına ölçekle
  layers.Conv2D(16, 3, padding='same', activation='relu'), # Evrişim katmanı 1
  layers.MaxPooling2D(), # Havuzlama katmanı 1
  layers.Conv2D(32, 3, padding='same', activation='relu'), # Evrişim katmanı 2
  layers.MaxPooling2D(), # Havuzlama katmanı 2
  layers.Conv2D(64, 3, padding='same', activation='relu'), # Evrişim katmanı 3
  layers.MaxPooling2D(), # Havuzlama katmanı 3
  layers.Conv2D(128, 3, padding='same', activation='relu'), # Evrişim katmanı 4
  layers.MaxPooling2D(), # Havuzlama katmanı 4
  layers.Dropout(0.25), # Overfitting'i azaltmak için Dropout katmanı
  layers.Flatten(), # Özellik haritalarını düzleştir
  layers.Dense(128, activation='relu'), # Tam bağlı katman (Dense)
  layers.Dense(1, activation='sigmoid') # Çıkış katmanı (ikili sınıflandırma için 1 nöron, sigmoid aktivasyonu)
])
```

*   `Sequential`: Katmanların sırayla eklendiği basit bir model türü.
*   `Rescaling(1./255)`: Resimlerin piksel değerlerini (0-255) 0 ile 1 arasına ölçekler. Bu, modelin daha stabil eğitilmesine yardımcı olur.
*   `Conv2D`: Resimlerdeki özellikleri (kenarlar, desenler vb.) öğrenen evrişim katmanları. Bu kodda, `padding='same'` kullanılması, giriş görüntüsünün boyutunun çıkış görüntüsü ile aynı olmasını sağlar. Bu, modelin daha derin katmanlar eklemesine olanak tanır ve bilgi kaybını azaltır.
*   `MaxPooling2D`: Özellik haritalarının boyutunu küçülterek hesaplama yükünü azaltır ve önemli özellikleri vurgular.
*   `Dropout`: Eğitim sırasında rastgele bazı nöronları devre dışı bırakarak modelin belirli nöronlara aşırı bağımlı olmasını engeller ve genelleme yeteneğini artırır.
*   `Flatten`: Çok boyutlu özellik haritalarını tek boyutlu bir vektöre dönüştürür.
*   `Dense`: Tam bağlı sinir ağı katmanları.
*   **Çıkış Katmanı**: İki sınıf (kedi/köpek) olduğu için tek bir nöron ve `sigmoid` aktivasyon fonksiyonu kullanıyoruz. Sigmoid, çıktıyı 0 ile 1 arasında bir olasılık değerine dönüştürür (örneğin, 0'a yakınsa 'kedi', 1'e yakınsa 'köpek').

### 7. Model Derleme

Eğitim sürecini yapılandırmak için modeli derliyoruz. Optimizasyon algoritmasını, kayıp fonksiyonunu ve değerlendirme metriklerini belirtiyoruz.

```python
# --- Model Derleme ---
print("Model derleniyor...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # İkili sınıflandırma için kayıp fonksiyonu
              metrics=['accuracy']) # Değerlendirme metriği: doğruluk

model.build((None, img_height, img_width, 3)) # Modeli build et (özellikle Data Augmentation varsa gerekli)
model.summary() # Modelin özetini (katmanlar, parametre sayısı) yazdır
```

*   `optimizer='adam'`: Ağırlıkları güncellemek için kullanılacak popüler bir optimizasyon algoritması.
*   `loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)`: İkili sınıflandırma problemleri için standart kayıp fonksiyonu. `from_logits=False` çünkü son katmanımız `sigmoid` aktivasyonu kullanıyor (modelin çıktısı, zaten 0 ile 1 arasında olduğu için kaybı hesaplamak için ek bir işlem yapılmasına gerek yoktur).
*   `metrics=['accuracy']`: Eğitim ve doğrulama sırasında modelin performansını izlemek için kullanılacak metrik (doğruluk oranı).
*   `model.build()`: Modelin girdi şeklini kesin olarak belirler. Data Augmentation katmanı gibi bazı katmanlar için bu gereklidir.
*   `model.summary()`: Modelin mimarisini ve katmanlardaki parametre sayılarını gösterir.

### 8. Model Eğitimi

Hazırlanan veri kümeleriyle modeli eğitiyoruz.

```python
# --- Model Eğitimi ---
print("Model eğitiliyor...")
history = model.fit(
  train_ds,           # Eğitim verisi
  validation_data=val_ds, # Doğrulama verisi
  epochs=epochs         # Belirlenen epoch sayısı kadar eğit
)
```

*   `model.fit()`: Asıl eğitim işlemini başlatan fonksiyondur.
*   `train_ds`: Eğitim için kullanılacak veri kümesi.
*   `validation_data=val_ds`: Her epoch sonunda modelin performansını değerlendirmek için kullanılacak doğrulama veri kümesi. Bu, modelin ezberleyip ezberlemediğini (overfitting) anlamamıza yardımcı olur.
*   `epochs`: Modelin veri seti üzerinden kaç kez geçeceği.
*   `history`: Eğitim süreci boyunca her epoch'taki kayıp (loss) ve doğruluk (accuracy) değerlerini içeren bir nesne döndürür.

### 9. Eğitim Sonuçlarını Görselleştirme

Eğitim ve doğrulama metriklerinin (doğruluk ve kayıp) epoch'lara göre değişimini çizdirerek modelin öğrenme sürecini görselleştiriyoruz.

```python
# --- Eğitim Sonuçlarını Görselleştirme ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

*   Bu kod, eğitim ve doğrulama setleri için doğruluk ve kayıp değerlerinin epoch'lara göre nasıl değiştiğini gösteren iki grafik çizer.
*   Eğitim doğruluğu artarken doğrulama doğruluğunun düşmeye başlaması veya sabit kalması, modelin ezberlemeye (overfitting) başladığının bir işareti olabilir.

### 10. Modeli Kaydetme

Eğitilen modeli daha sonra kullanmak üzere dosyaya kaydediyoruz.

```python
# --- Modeli Kaydetme ---
print(f"Model '{model_save_path}' olarak kaydediliyor...")
model.save(model_save_path) # Modeli belirtilen yola kaydet
print("Model başarıyla kaydedildi.")

print("Eğitim tamamlandı.")
```

*   `model.save()`: Modelin mimarisini, ağırlıklarını ve eğitim konfigürasyonunu (varsa optimizer durumu) belirtilen dosyaya kaydeder.

## Bölüm 2: Eğitilmiş Modeli Kullanma ve Yanlış Tahminlerle Yeniden Eğitim

Bu bölümde, `Bölüm 1`'de eğittiğimiz modeli (`cat_dog_classifier.keras`) yükleyip yeni bir resim üzerinde tahmin yapacağız. Eğer tahmin yanlışsa, modeli bu tek resimle yeniden eğiterek (fine-tuning benzeri bir yaklaşımla) performansını iyileştirmeye çalışacağız.

### 1. Gerekli Kütüphaneler

Tahmin ve yeniden eğitim için gerekli kütüphaneleri içe aktarıyoruz.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tensorflow.keras.preprocessing import image
```

*   `numpy`: Sayısal işlemler, özellikle etiketleri modele uygun formata getirmek için kullanılır.
*   `tensorflow.keras.preprocessing.image`: Resimleri yüklemek ve ön işlemek için yardımcı fonksiyonlar içerir.

### 2. Parametreler ve Model Yükleme

Gerekli parametreleri tanımlayıp önceden eğittiğimiz modeli yüklüyoruz.

```python
# --- Parametreler ---
img_height = 180
img_width = 180
model_path = 'cat_dog_classifier.keras'
# Sınıf isimleri Bölüm 1'deki ile aynı sırada olmalı!
class_names = ['cat', 'dog']

# --- Modeli Yükle ---
print(f"Model yükleniyor: {model_path}")
try:
    model = keras.models.load_model(model_path)
    print("Model başarıyla yüklendi.")
    model.summary() # Yüklenen modelin özetini göster
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()
```

*   `img_height`, `img_width`: Eğitim sırasında kullanılan boyutlarla aynı olmalıdır.
*   `model_path`: Eğitilmiş modelin dosya yolu.
*   `class_names`: Sınıf isimleri, `Bölüm 1`'deki `image_dataset_from_directory` tarafından bulunan sıra ile (genellikle alfabetik) aynı olmalıdır. Bu, tahmin edilen indeksin doğru sınıfa eşlenmesi için kritiktir.
*   `keras.models.load_model()`: Kaydedilmiş Keras modelini yükler.

### 3. Görüntü İşleme Fonksiyonu

Tahmin veya yeniden eğitim için verilecek tek bir görüntüyü yükleyip modelin beklediği formata (doğru boyut, batch boyutu) getiren bir fonksiyon tanımlıyoruz.

```python
# --- Görüntü İşleme Fonksiyonu ---
def preprocess_image(img_path):
    """Verilen yoldaki görüntüyü yükler ve model için ön işler."""
    try:
        img = image.load_img(img_path, target_size=(img_height, img_width)) # Resmi yükle ve boyutlandır
        img_array = image.img_to_array(img) # Resmi NumPy dizisine çevir
        img_array = tf.expand_dims(img_array, 0) # Batch boyutu ekle (shape: (1, height, width, channels))
        # ÖNEMLİ: Bölüm 1'de Rescaling(1./255) katmanı modelin içinde olduğu için
        # burada tekrar ölçekleme YAPMIYORUZ. Model zaten bunu yapacak.
        return img_array
    except FileNotFoundError:
        print(f"Hata: Görüntü dosyası bulunamadı: {img_path}")
        return None
    except Exception as e:
        print(f"Görüntü işlenirken hata oluştu: {e}")
        return None
```

*   `image.load_img()`: Resmi belirtilen boyutta yükler.
*   `image.img_to_array()`: Resmi bir NumPy dizisine dönüştürür.
*   `tf.expand_dims()`: Modele tek bir resim versek bile, model genellikle bir batch (toplu iş) bekler. Bu fonksiyon, diziye bir batch boyutu (1) ekler.

### 4. Tahmin Fonksiyonu

Ön işlenmiş bir görüntü üzerinde modelin tahminini yapan bir fonksiyon tanımlıyoruz.

```python
# --- Tahmin Fonksiyonu ---
def predict_image(img_path):
    """Bir görüntünün sınıfını tahmin eder."""
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return None, None

    predictions = model.predict(processed_img) # Modeli kullanarak tahmini yap
    score = predictions[0][0] # Sigmoid çıktısı (tek nöron olduğu için)

    # Skoru sınıfa çevir (0.5 eşik değeri)
    predicted_class_index = 1 if score > 0.5 else 0
    predicted_class_name = class_names[predicted_class_index]
    # Güven skorunu hesapla
    confidence = score if predicted_class_index == 1 else 1 - score

    print(
        f"Bu görüntü {confidence:.2f} güvenle bir {predicted_class_name}."
    )
    return predicted_class_name, confidence
```

*   `model.predict()`: Verilen girdi için modelin çıktısını hesaplar. Sigmoid aktivasyonlu tek nöronlu modelimiz için bu, 0 ile 1 arasında tek bir değerdir.
*   `score = predictions[0][0]`: Tahmin sonucu genellikle `[[değer]]` şeklinde bir dizi içinde gelir, bu yüzden skoru almak için `[0][0]` kullanırız.
*   Eşik Değeri (0.5): Skor 0.5'ten büyükse ikinci sınıfa ('dog'), küçükse veya eşitse ilk sınıfa ('cat') ait olduğunu varsayarız.
*   `confidence`: Modelin tahminine ne kadar güvendiğini gösterir (0.5'ten uzaklık).

### 5. Yeniden Eğitim Fonksiyonu

Modelin yanlış tahmin ettiği bir görüntü ile modeli yeniden eğitmek (ağırlıkları güncellemek) için bir fonksiyon tanımlıyoruz.

```python
# --- Yeniden Eğitim Fonksiyonu ---
def retrain_model_with_image(img_path, true_label_index):
    """Modeli yanlış tahmin edilen bir görüntü ile yeniden eğitir."""
    print(f"\n'{img_path}' görüntüsü yanlış tahmin edildi. Model yeniden eğitiliyor...")
    print(f"Doğru etiket: {class_names[true_label_index]}")

    processed_img = preprocess_image(img_path)
    if processed_img is None:
        print("Görüntü işlenemediği için yeniden eğitim yapılamıyor.")
        return False

    # Doğru etiketi modelin beklediği formata getir (shape: (1, 1))
    label = np.array([[float(true_label_index)]]) # BinaryCrossentropy için float

    # Modeli tek örnekle kısa bir süre eğit (NumPy array'lerini doğrudan kullan)
    # model.fit, tf.data.Dataset yerine NumPy array'lerini de kabul eder.
    print("Yeniden eğitim başlıyor...")
    # Sadece birkaç epoch eğitmek genellikle yeterlidir.
    # verbose=0, eğitim ilerlemesini yazdırmaz.
    history = model.fit(processed_img, label, epochs=5, verbose=0)
    print("Yeniden eğitim tamamlandı.")

    # İsteğe bağlı: Yeniden eğitim sonrası doğruluğu kontrol et
    loss, accuracy = model.evaluate(processed_img, label, verbose=0)
    print(f"Yeniden eğitim sonrası doğruluk (tek örnek üzerinde): {accuracy:.2f}")

    return True
```

*   Bu fonksiyon, yanlış tahmin edilen resmin yolunu (`img_path`) ve doğru etiket indeksini (`true_label_index`, 'cat' için 0, 'dog' için 1) alır.
*   Resmi `preprocess_image` ile işler.
*   Doğru etiketi `[[0.]]` veya `[[1.]]` şeklinde bir NumPy dizisine dönüştürür.
*   `model.fit()` fonksiyonunu **sadece bu tek resim ve etiketle** çağırır. Bu, modelin ağırlıklarını bu spesifik örneğe göre hafifçe ayarlamasını sağlar.
*   `epochs=5`: Modeli bu tek örnek üzerinde 5 kez eğitiriz. Bu, öğrenmeyi pekiştirmeye yardımcı olabilir, ancak çok fazla epoch ezberlemeye yol açabilir.
*   `model.evaluate()`: Yeniden eğitimin hemen ardından modelin aynı örnek üzerindeki performansını kontrol ederiz (genellikle doğruluk 1.0 olmalıdır).

### 6. Ana Çalıştırma Bloğu

Script'i çalıştırdığımızda gerçekleşecek ana mantığı içerir: bir test görüntüsü seçer, tahmin yapar, tahmin yanlışsa yeniden eğitir ve güncellenmiş modeli kaydeder.

```python
# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    # Test edilecek görüntü ve doğru etiketi belirle
    test_image_path = 'dataset/cat/Image_1.jpg' # Örnek bir kedi resmi
    true_label = 'cat' # Bu resmin doğru etiketi

    # VEYA bir köpek resmi dene:
    # test_image_path = 'dataset/dog/Image_1.jpg'
    # true_label = 'dog'

    if not os.path.exists(test_image_path):
        print(f"Hata: Test görüntüsü bulunamadı: {test_image_path}")
    else:
        # Tahmin yap
        predicted_label, confidence = predict_image(test_image_path)

        if predicted_label: # Tahmin başarılı olduysa
            true_label_index = class_names.index(true_label)
            predicted_label_index = class_names.index(predicted_label)

            # Tahmin yanlış mı kontrol et
            if predicted_label != true_label:
                print(f"Tahmin YANLIŞ! (Tahmin: {predicted_label}, Gerçek: {true_label})")
                # Yeniden eğitim fonksiyonunu çağır
                retrained = retrain_model_with_image(test_image_path, true_label_index)
                if retrained:
                    # Güncellenmiş modeli kaydet
                    print(f"Güncellenmiş model kaydediliyor: {model_path}")
                    model.save(model_path)
                    print("Güncellenmiş model başarıyla kaydedildi.")

                    # Yeniden eğitim sonrası tekrar tahmin et (isteğe bağlı)
                    print("\nYeniden eğitim sonrası tekrar tahmin ediliyor:")
                    predict_image(test_image_path)
            else:
                # Tahmin doğruysa
                print(f"Tahmin DOĞRU! (Tahmin: {predicted_label}, Gerçek: {true_label})")
        else:
            print("Görüntü işlenemediği için tahmin yapılamadı.")

```

*   Bu blok, `test_image_path` ile belirtilen resim için `predict_image` fonksiyonunu çağırır.
*   Tahmin edilen etiket (`predicted_label`) ile gerçek etiket (`true_label`) karşılaştırılır.
*   Eğer farklıysa, `retrain_model_with_image` fonksiyonu çağrılır.
*   Yeniden eğitim başarılı olursa (`retrained` True dönerse), güncellenmiş model `model.save()` ile tekrar aynı dosyaya kaydedilir.
*   İsteğe bağlı olarak, yeniden eğitim sonrası modelin aynı resim üzerindeki tahmini tekrar yazdırılır (genellikle artık doğru tahmin etmesi beklenir).

# Sonuç

Bu eğitim, temel bir resim sınıflandırma modelinin nasıl oluşturulacağını ve eğitilmiş bir modelin yeni, zorlayıcı örneklerle nasıl güncellenebileceğini göstermektedir. 
    - **Sıfırdan Model Eğitimi**: Eğitim sürecinde, sıfırdan bir Evrişimli Sinir Ağı (CNN) modeli oluşturduk ve bu modeli kedi ve köpek resimlerini sınıflandırmak için eğittik. Modelin mimarisi, veri artırma teknikleri ve performans optimizasyonu gibi önemli adımları içeriyordu.
    - **Modelin Yeniden Eğitimi**: Eğitilmiş modelin yanlış tahmin yaptığı durumlarda, bu örnekleri kullanarak modeli yeniden eğitme sürecini inceledik. Bu yöntem, modelin yeni verilere adapte olmasını sağladı.

- **Transfer Learning Nedir?**
    - Transfer learning, önceden eğitilmiş bir modelin ağırlıklarını kullanarak yeni bir probleme adapte etme yöntemidir. Bu yaklaşım, özellikle sınırlı veri setleriyle çalışırken oldukça etkilidir.
    - Bu eğitimde, transfer learning konseptine benzer bir şekilde, sıfırdan bir model oluşturmanın yanı sıra, eğitilmiş bir modeli yeni verilerle yeniden eğiterek performansını artırmayı öğrendik.

- **Bu Yöntemin Avantajları**
    - **Zaman Kazandırır**: Sıfırdan bir model eğitmek yerine, mevcut bir modeli yeniden eğitmek daha hızlıdır.
    - **Daha İyi Genelleştirme**: Transfer learning veya yeniden eğitim, modelin daha az veriyle daha iyi sonuçlar vermesini sağlar.
    - **Esneklik**: Model, yeni ve zorlayıcı örneklerle kolayca güncellenebilir.

Bu yöntemler, hem derin öğrenme modellerinin geliştirilmesinde hem de gerçek dünya problemlerine hızlı çözümler üretmede oldukça etkili araçlardır.
