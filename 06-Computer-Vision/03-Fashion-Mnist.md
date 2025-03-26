# Fashion MNIST ile Görüntü Sınıflandırma

Bu eğitimde, Fashion MNIST veri setini kullanarak basit bir Konvolüsyonel Sinir Ağı (CNN) modeli oluşturacağız ve eğiteceğiz. Fashion MNIST, giysi görüntülerinden oluşan bir veri setidir ve her görüntü bir sınıfa aittir.

## Gerekli Kütüphanelerin Yüklenmesi

İlk olarak, gerekli kütüphaneleri yükleyelim.

```python
# Gerekli kütüphanelerin yüklenmesi
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```

## Veri Setinin Yüklenmesi

Fashion MNIST veri setini yükleyelim ve sınıf isimlerini tanımlayalım.

```python
# Fashion MNIST veri setinin yüklenmesi
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Sınıf isimlerinin tanımlanması
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## Örnek Görüntülerin Görselleştirilmesi

Veri setinden bazı örnek görüntüleri görselleştirelim.

```python
# Veri setinden örnek görüntülerin görselleştirilmesi
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

## Veri Ön İşleme

Görüntülerin piksel değerlerini 0-1 aralığına ölçekleyerek normalizasyon yapalım.

```python
# Veri ön işleme: Piksel değerlerinin 0-1 aralığına ölçeklenmesi
train_images = train_images / 255.0
test_images = test_images / 255.0
```

## CNN Modelinin Oluşturulması

CNN modelimizi oluşturalım. Modelimizde konvolüsyon ve havuzlama katmanları kullanacağız.

```python
# CNN modelinin oluşturulması
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
```

- **Conv2D Katmanı (32 filtre, 3x3 boyutunda)**: Bu katman, giriş görüntüsünden 32 farklı özellik haritası çıkarır. Her bir filtre, 3x3 boyutunda olup, görüntü üzerinde kaydırılarak belirli özellikleri (kenarlar, köşeler vb.) öğrenir. `relu` aktivasyon fonksiyonu, negatif değerleri sıfıra çevirerek doğrusal olmayanlık ekler.
  
- **MaxPooling2D Katmanı (2x2 boyutunda)**: Bu katman, özellik haritalarının boyutunu 2x2 pencereler kullanarak yarıya indirir. Her pencere içindeki en yüksek değeri seçer. Bu işlem, modelin hesaplama maliyetini azaltır ve konum değişikliklerine karşı daha dayanıklı olmasını sağlar.

- **Conv2D Katmanı (64 filtre, 3x3 boyutunda)**: İkinci konvolüsyon katmanı, daha karmaşık özellikleri öğrenmek için 64 filtre kullanır. Bu katman, ilk konvolüsyon katmanından gelen özellik haritalarını alır ve daha derin özellikler çıkarır.

- **MaxPooling2D Katmanı (2x2 boyutunda)**: İkinci havuzlama katmanı, yine 2x2 pencereler kullanarak özellik haritalarının boyutunu yarıya indirir.

- **Conv2D Katmanı (64 filtre, 3x3 boyutunda)**: Üçüncü konvolüsyon katmanı, daha da karmaşık özellikleri öğrenmek için 64 filtre kullanır. Bu katman, önceki katmanlardan gelen özellik haritalarını alır ve daha yüksek seviyeli özellikler çıkarır.

- **Flatten Katmanı**: Bu katman, çok boyutlu özellik haritalarını tek boyutlu bir vektöre dönüştürür. Bu işlem, tam bağlantılı katmanlara giriş olarak kullanılacak veriyi hazırlar.

- **Dense Katmanı (64 nöron)**: Bu tam bağlantılı katman, 64 nöron içerir ve `relu` aktivasyon fonksiyonunu kullanır. Bu katman, görüntüdeki yüksek seviyeli özellikleri öğrenir.

- **Dense Katmanı (10 nöron)**: Çıkış katmanı, 10 nöron içerir ve her nöron bir sınıfı temsil eder. Bu katman, sınıflandırma görevini gerçekleştirir.

## Modelin Derlenmesi

Modelimizi derleyelim. Optimizasyon için Adam algoritmasını ve kayıp fonksiyonu olarak Sparse Categorical Crossentropy'yi kullanacağız.

```python
# Modelin derlenmesi
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

## Modelin Eğitilmesi

Modelimizi eğitelim ve eğitim sürecini doğrulama verileri ile izleyelim.

```python
# Modelin eğitilmesi
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```

## Eğitim ve Doğrulama Sonuçlarının Görselleştirilmesi

Eğitim ve doğrulama kayıplarını ve doğruluklarını görselleştirelim.

```python
# Eğitim ve doğrulama kayıplarının ve doğruluklarının görselleştirilmesi
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

## Modelin Değerlendirilmesi

Modelimizi test veri seti üzerinde değerlendirelim ve doğruluğunu görelim.

```python
# Test veri seti üzerinde modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest doğruluğu: {test_acc}')
```

## Sonuç

Bu projede, Fashion MNIST veri seti üzerinde bir CNN modeli geliştirdik. Modelimiz, giysi görüntülerini başarılı bir şekilde sınıflandırabildi. Konvolüsyonel katmanlar sayesinde görüntülerdeki önemli özellikleri öğrenerek yüksek doğruluk oranlarına ulaştık. Bu çalışma, derin öğrenme modellerinin görüntü sınıflandırma problemlerindeki etkinliğini göstermektedir.