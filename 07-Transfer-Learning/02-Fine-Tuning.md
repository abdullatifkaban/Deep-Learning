# Transfer Learning ve Fine-tuning ile Resim Sınıflandırma (MobileNetV2)

Bu eğitimde, önceden eğitilmiş bir modeli (MobileNetV2) kullanarak transfer learning ve fine-tuning teknikleriyle kedi ve köpek resimlerini sınıflandıran bir model oluşturacağız. Bu yaklaşım, sıfırdan model eğitmekten genellikle daha hızlıdır ve daha az veriyle daha yüksek doğruluk oranları elde etmeyi sağlar.

## 1. Gerekli Kütüphanelerin İçe Aktarılması

Projemiz için gerekli olan TensorFlow, Keras ve diğer yardımcı kütüphaneleri içe aktarıyoruz.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib
matplotlib.use('Agg') # Grafik çizimi için arka planı ayarla
import matplotlib.pyplot as plt
import numpy as np
```

*   `tensorflow`, `keras`, `layers`: Derin öğrenme modeli oluşturma, yükleme ve eğitme için temel araçlar.
*   `os`: Dosya yolları gibi işletim sistemi işlemleri için.
*   `matplotlib`: Eğitim sonuçlarını görselleştirmek için. `matplotlib.use('Agg')` komutu, grafiklerin sunucu gibi grafik arayüzü olmayan ortamlarda da kaydedilebilmesini sağlar.
*   `numpy`: Sayısal işlemler için.

## 2. Parametrelerin Tanımlanması

Model eğitimi, veri işleme ve fine-tuning süreci için gerekli parametreleri belirliyoruz.

```python
# --- Parametreler ---
img_height = 160 # MobileNetV2 genellikle 160x160 veya 224x224 ile iyi çalışır
img_width = 160
batch_size = 32
data_dir = 'dataset'
model_save_path = 'cat_dog_finetuned_mobilenetv2.keras'
initial_epochs = 10 # Sadece yeni katmanları eğitmek için epoch sayısı
fine_tune_epochs = 10 # Fine-tuning için ek epoch sayısı
total_epochs = initial_epochs + fine_tune_epochs
base_learning_rate = 0.001 # Başlangıç öğrenme oranı
fine_tune_learning_rate = base_learning_rate / 10 # Fine-tuning için daha düşük öğrenme oranı
```

*   `img_height`, `img_width`: Kullanacağımız önceden eğitilmiş modelin (MobileNetV2) genellikle beklediği girdi boyutlarından biri (160x160).
*   `batch_size`: Her adımda işlenecek resim sayısı.
*   `data_dir`: Veri setinin bulunduğu klasör.
*   `model_save_path`: Sonuçta elde edilecek fine-tuned modelin kaydedileceği dosya adı.
*   `initial_epochs`: Transfer learning aşamasında (sadece yeni eklenen katmanları eğitirken) kullanılacak epoch sayısı.
*   `fine_tune_epochs`: Fine-tuning aşamasında (temel modelin bazı katmanları çözüldükten sonra) kullanılacak ek epoch sayısı.
*   `total_epochs`: Toplam eğitim süresi.
*   `base_learning_rate`: Yeni katmanları eğitirken kullanılacak öğrenme oranı.
*   `fine_tune_learning_rate`: Fine-tuning aşamasında kullanılacak daha düşük öğrenme oranı. Bu, önceden öğrenilmiş özelliklerin bozulmasını önlemek için önemlidir.

## 3. Veri Yükleme ve Hazırlama

`image_dataset_from_directory` kullanarak resimleri diskten yüklüyor, eğitim ve doğrulama setlerine ayırıyoruz.

```python
# --- Veri Yükleme ve Hazırlama ---
print("Veri yükleniyor...")
# Eğitim veri kümesi
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Doğrulama veri kümesi
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print("Sınıflar:", class_names)
num_classes = len(class_names)

# Performans için veri kümelerini yapılandırma
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

*   Bu kısım, `train_model.py`'deki veri yükleme adımıyla benzerdir. Veriler yüklenir, ayrılır ve performans için optimize edilir (`cache`, `shuffle`, `prefetch`).

## 4. Ön İşleme ve Veri Artırma

Önceden eğitilmiş modeller genellikle belirli bir ön işleme adımı bekler. MobileNetV2, piksel değerlerinin -1 ile 1 arasında olmasını bekler. Ayrıca, veri artırma (data augmentation) uyguluyoruz.

```python
# --- MobileNetV2 Ön İşleme Katmanı ---
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Veri artırma (Data Augmentation) - Fine-tuning için de faydalı
data_augmentation = keras.Sequential([
  layers.RandomFlip('horizontal'),
  layers.RandomRotation(0.2),
])
```

*   `tf.keras.applications.mobilenet_v2.preprocess_input`: MobileNetV2 için gerekli ön işleme adımlarını (piksel değerlerini -1 ile 1 arasına ölçekleme vb.) içeren fonksiyondur.
*   `data_augmentation`: Eğitim sırasında resimlere rastgele çevirme ve döndürme uygulayarak modelin genelleme yeteneğini artırır.

## 5. Önceden Eğitilmiş Modeli Yükleme (MobileNetV2)

ImageNet veri seti üzerinde eğitilmiş MobileNetV2 modelini, kendi sınıflandırma katmanı olmadan (`include_top=False`) yüklüyoruz.

```python
# --- Önceden Eğitilmiş Modeli Yükleme (MobileNetV2) ---
print("Önceden eğitilmiş MobileNetV2 modeli yükleniyor...")
IMG_SHAPE = (img_height, img_width, 3)

# include_top=False: ImageNet için olan sınıflandırma katmanını dahil etme
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet') # ImageNet ağırlıklarını kullan
```

*   `tf.keras.applications.MobileNetV2`: Keras içinde hazır bulunan MobileNetV2 modelini yükler.
*   `input_shape`: Modelin beklediği girdi resim boyutunu belirtir.
*   `include_top=False`: Modelin orijinal (ImageNet için 1000 sınıfı tahmin eden) sınıflandırma katmanını yüklemememizi sağlar. Biz kendi sınıflandırıcımızı ekleyeceğiz.
*   `weights='imagenet'`: Modelin ImageNet üzerinde eğitilmiş ağırlıklarla yüklenmesini sağlar. Bu ağırlıklar, genel görsel özellikleri (kenarlar, dokular vb.) öğrenmiştir.

## 6. Temel Model Katmanlarını Dondurma

Transfer learning'in ilk aşamasında, temel modelin (MobileNetV2) öğrenmiş olduğu özellikleri korumak isteriz. Bu nedenle, katmanlarının ağırlıklarının eğitim sırasında güncellenmesini engellemek için `trainable` özelliğini `False` yaparız.

```python
# --- Temel Model Katmanlarını Dondurma ---
base_model.trainable = False
print(f"Temel modeldeki katman sayısı: {len(base_model.layers)}")
```

*   `base_model.trainable = False`: Bu komut, `base_model` içindeki tüm katmanların ağırlıklarını dondurur. Eğitim sırasında sadece sonradan ekleyeceğimiz katmanların ağırlıkları güncellenecektir.

## 7. Yeni Sınıflandırma Katmanları Ekleme

Dondurulmuş temel modelin çıktısını alıp kendi problemimize (kedi/köpek sınıflandırma) uygun yeni bir sınıflandırma başlığı ekliyoruz.

```python
# --- Yeni Sınıflandırma Katmanları Ekleme ---
inputs = tf.keras.Input(shape=IMG_SHAPE) # Modelin girdi katmanı
x = data_augmentation(inputs) # Önce veri artırma uygula
x = preprocess_input(x)       # Sonra MobileNetV2 ön işlemesini yap
x = base_model(x, training=False) # Temel modeli çalıştır (training=False önemli!)
x = layers.GlobalAveragePooling2D()(x) # Özellik haritalarını tek bir vektöre indirge
x = layers.Dropout(0.2)(x) # Overfitting'i azaltmak için Dropout
outputs = layers.Dense(1, activation='sigmoid')(x) # İkili sınıflandırma için çıkış katmanı

model = tf.keras.Model(inputs, outputs) # Yeni modeli oluştur
```

*   `tf.keras.Input`: Modelin girdi katmanını tanımlar.
*   `data_augmentation(inputs)`: Girdiye veri artırma uygulanır.
*   `preprocess_input(x)`: Veri artırılmış resimlere MobileNetV2 ön işlemesi uygulanır.
*   `base_model(x, training=False)`: Ön işlenmiş veri dondurulmuş temel modele verilir. `training=False` parametresi, temel modeldeki `BatchNormalization` gibi katmanların çıkarım (inference) modunda çalışmasını sağlar, bu aşamada bu önemlidir.
*   `layers.GlobalAveragePooling2D()`: Temel modelin çıktısı olan özellik haritalarının (örneğin 5x5x1280) her bir kanalının ortalamasını alarak tek bir vektöre (1280 elemanlı) dönüştürür. Bu, `Flatten` katmanına göre daha az parametre oluşturur ve genellikle overfitting'i azaltır.
*   `layers.Dropout(0.2)`: Rastgele nöronları sıfırlayarak ezberlemeyi azaltır.
*   `layers.Dense(1, activation='sigmoid')`: İkili sınıflandırma için tek nöronlu ve sigmoid aktivasyonlu son katmanımız.
*   `tf.keras.Model(inputs, outputs)`: Girdi katmanını ve çıktı katmanını birbirine bağlayarak yeni, tam modelimizi oluşturur.

## 8. Modeli Derleme (İlk Aşama - Transfer Learning)

Modeli, sadece yeni eklenen katmanların eğitileceği ilk aşama için derliyoruz.

```python
# --- Modeli Derleme (İlk Aşama) ---
print("Model ilk aşama için derleniyor (sadece yeni katmanlar eğitilecek)...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary() # Tüm modelin özetini göster
```

*   `optimizer`: Adam optimizörü kullanılır.
*   `loss`: İkili sınıflandırma için `BinaryCrossentropy`.
*   `metrics`: Doğruluk metriği izlenir.
*   `model.summary()`: Bu aşamada, temel modelin parametrelerinin "Non-trainable params" olarak göründüğüne dikkat edin. Sadece `GlobalAveragePooling2D`, `Dropout` ve `Dense` katmanlarının parametreleri eğitilecektir.

## 9. Yeni Katmanları Eğitme

Modeli, sadece yeni eklenen sınıflandırma katmanlarını eğitmek üzere `initial_epochs` kadar çalıştırıyoruz.

```python
# --- Yeni Katmanları Eğitme ---
print(f"Yeni sınıflandırma katmanları {initial_epochs} epoch boyunca eğitiliyor...")
history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)

# --- Eğitim Geçmişini Kaydet (İlk Aşama) ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
```

*   Bu `model.fit` çağrısı, sadece `GlobalAveragePooling2D` ve `Dense` katmanlarının ağırlıklarını günceller, çünkü `base_model.trainable` hala `False`'tur.
*   Eğitim geçmişi (doğruluk ve kayıp değerleri) `history` nesnesinde saklanır ve daha sonra görselleştirme için değişkenlere atanır.

## 10. Fine-tuning Aşaması

Bu aşamada, temel modelin (MobileNetV2) üst katmanlarından bazılarını "çözeriz" (eğitilebilir hale getiririz) ve modeli daha düşük bir öğrenme oranıyla eğitmeye devam ederiz. Bu, önceden öğrenilmiş genel özelliklerin, bizim özel veri setimize (kedi/köpek) daha iyi uyum sağlamasına olanak tanır.

### 10.1. Temel Modelin Üst Katmanlarını Çözme

Önce tüm temel modeli eğitilebilir hale getiririz, sonra belirli bir katmandan önceki katmanları tekrar dondururuz.

```python
# 1. Temel Modelin Üst Katmanlarını Çözme
base_model.trainable = True # Önce tüm temel modeli çöz
print(f"\nTemel modeldeki katman sayısı: {len(base_model.layers)}")

# Hangi katmandan itibaren çözüleceğini belirleyelim
fine_tune_at = 100 # İlk 100 katmanı dondurulmuş bırakalım

# İlk 'fine_tune_at' katmanını dondur
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

print(f"Temel modelin ilk {fine_tune_at} katmanı donduruldu, geri kalanı eğitilebilir.")
```

*   `base_model.trainable = True`: Tüm temel model katmanlarını eğitilebilir yapar.
*   `fine_tune_at`: Hangi katmandan sonrasının eğitileceğini belirler. Daha derin (girdiye daha yakın) katmanlar genellikle daha genel özellikleri (kenarlar, renkler) öğrenir. Daha sığ (çıktıya daha yakın) katmanlar ise daha karmaşık, probleme özgü özellikleri öğrenir. Fine-tuning'de genellikle bu sığ katmanları eğitiriz.
*   `for layer in base_model.layers[:fine_tune_at]: layer.trainable = False`: İlk `fine_tune_at` kadar katmanı tekrar dondurur. Böylece sadece `fine_tune_at`'den sonraki katmanlar ve bizim eklediğimiz başlık eğitilebilir olur.

### 10.2. Modeli Daha Düşük Öğrenme Oranıyla Yeniden Derleme

Fine-tuning yaparken, önceden öğrenilmiş ağırlıkları çok fazla bozmamak için genellikle çok daha düşük bir öğrenme oranı kullanılır. Modeli bu yeni öğrenme oranıyla yeniden derleriz.

```python
# 2. Modeli Daha Düşük Öğrenme Oranıyla Yeniden Derleme
print("Model fine-tuning için yeniden derleniyor...")
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=fine_tune_learning_rate), # Düşük öğrenme oranı!
              metrics=['accuracy'])

model.summary() # Güncellenmiş eğitilebilir parametreleri göster
print(f"Toplam eğitilebilir parametre sayısı: {len(model.trainable_variables)}")
```

*   `optimizer = tf.keras.optimizers.RMSprop(learning_rate=fine_tune_learning_rate)`: Fine-tuning için sıklıkla RMSprop veya SGD gibi optimizörler düşük öğrenme oranlarıyla kullanılır.
*   `model.summary()`: Bu kez, temel modelin son katmanlarının da "Trainable params" kısmında yer aldığını göreceksiniz.

### 10.3. Fine-tuning Eğitimi

Modeli, çözülmüş temel katmanlar ve yeni başlıkla birlikte, `fine_tune_epochs` kadar daha eğitiyoruz. Eğitime kaldığımız yerden devam etmek için `initial_epoch` parametresini kullanırız.

```python
# 3. Fine-tuning Eğitimi
print(f"Fine-tuning {fine_tune_epochs} epoch boyunca devam ediyor...")
history_fine = model.fit(train_ds,
                         epochs=total_epochs, # Toplam epoch sayısına kadar devam et
                         initial_epoch=history.epoch[-1] + 1, # Kaldığı yerden devam et
                         validation_data=val_ds)
```

*   `epochs=total_epochs`: Eğitimi toplam epoch sayısına kadar sürdürür.
*   `initial_epoch=history.epoch[-1] + 1`: Eğitimin önceki aşamanın bittiği epoch'tan devam etmesini sağlar. Bu, öğrenme eğrilerinin ve logların doğru tutulması için önemlidir.

## 11. Sonuçları Görselleştirme ve Modeli Kaydetme

Her iki eğitim aşamasının (transfer learning ve fine-tuning) geçmişini birleştirip doğruluk ve kayıp grafiklerini çizdiririz. Fine-tuning'in başladığı noktayı grafikte işaretleriz. Son olarak, tamamen eğitilmiş modeli kaydederiz.

```python
# --- Tüm Eğitim Geçmişini Birleştirme ve Görselleştirme ---
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(12, 8))
# ... (Grafik çizdirme kodları) ...
plt.savefig('fine_tuning_history.png')
print("Fine-tuning eğitim grafiği 'fine_tuning_history.png' olarak kaydedildi.")

# --- Fine-tuned Modeli Kaydetme ---
print(f"Fine-tuned model '{model_save_path}' olarak kaydediliyor...")
model.save(model_save_path)
print("Model başarıyla kaydedildi.")

print("Fine-tuning tamamlandı.")
```

*   İlk aşamanın (`history`) ve fine-tuning aşamasının (`history_fine`) metrikleri birleştirilir.
*   Matplotlib kullanılarak birleşik doğruluk ve kayıp grafikleri oluşturulur ve `fine_tuning_history.png` olarak kaydedilir. Grafikte fine-tuning'in başladığı epoch dikey bir çizgiyle belirtilir.
*   `model.save()`: Son, fine-tuned model dosyaya kaydedilir.

# Sonuç
Bu eğitim, transfer learning ve fine-tuning tekniklerini kullanarak mevcut güçlü modellerden nasıl yararlanılacağını ve bunları kendi özel problemlerimize nasıl uyarlayabileceğimizi göstermektedir. Önceden eğitilmiş bir modelin genel özelliklerini kullanarak sıfırdan model eğitmenin zorluklarını aşabilir ve daha az veriyle daha yüksek doğruluk oranları elde edebiliriz. Ayrıca, fine-tuning aşamasıyla modelin belirli bir probleme daha iyi uyum sağlamasını sağlayarak performansı artırabiliriz. Bu süreç, derin öğrenme projelerinde zaman ve kaynak tasarrufu sağlarken, aynı zamanda daha etkili sonuçlar elde etmemize olanak tanır.

# Bonus

`ResNet50` modelini kullanarak resim tahmin edelim.

```py
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from IPython.display import Image

model =  ResNet50(weights="imagenet")

img=image.load_img("dog.jpeg", target_size=(224,224))
img=image.img_to_array(img)
img=np.expand_dims(img, axis=0)
img=preprocess_input(img)

pred = model.predict(img)
predictions = decode_predictions(pred, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
```

- `img_height`, `img_width`: Kullanacağımız önceden eğitilmiş modelin (MobileNetV2) genellikle beklediği girdi boyutlarından biri (160x160).
- `batch_size`: Her adımda işlenecek resim sayısı.
- `data_dir`: Veri setinin bulunduğu klasör.
- `model_save_path`: Sonuçta elde edilecek fine-tuned modelin kaydedileceği dosya adı.
- `initial_epochs`: Transfer learning aşamasında (sadece yeni eklenen katmanları eğitirken) kullanılacak epoch sayısı.
- `fine_tune_epochs`: Fine-tuning aşamasında (temel modelin bazı katmanları çözüldükten sonra) kullanılacak ek epoch sayısı.
- `total_epochs`: Toplam eğitim süresi.
- `base_learning_rate`: Yeni katmanları eğitirken kullanılacak öğrenme oranı.
- `fine_tune_learning_rate`: Fine-tuning aşamasında kullanılacak daha düşük öğrenme oranı. Bu, önceden öğrenilmiş özelliklerin bozulmasını önlemek için önemlidir.