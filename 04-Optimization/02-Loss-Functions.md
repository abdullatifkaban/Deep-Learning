# Kayıp Fonksiyonları (Loss Functions)

Loss fonksiyonları, derin öğrenme modellerinin başarısını değerlendirmede ve optimize etmede kullanılan temel matematik araçlarıdır. Bu fonksiyonlar, modelin tahminleri ile gerçek değerler arasındaki sapmanın nicel bir ölçüsünü sağlar ve modelin eğitimi sırasında bu sapmanın minimize edilmesine rehberlik eder.

## 1. Categorical Cross-Entropy
**Çok sınıflı sınıflandırma** problemlerinin standardı olan bu fonksiyon, görüntü sınıflandırma, duygu analizi, nesne tanıma gibi birden fazla sınıf içeren problemlerde kullanılır. Her bir sınıf için olasılık dağılımı üreterek, doğru sınıfın olasılığını maksimize etmeye çalışır.

```python
import tensorflow as tf

loss = tf.keras.losses.CategoricalCrossentropy()
```

### 🔍 Özellikleri:
- One-hot encoded etiketlerle çalışır (örn: [0,1,0] şeklinde)
- Softmax aktivasyon fonksiyonu ile birlikte optimum performans gösterir
- Sınıflar arası dengeli bir öğrenme sağlar
- Gradyan patlaması/sönmesi problemlerine karşı dirençlidir

## 2. Binary Cross-Entropy
**İkili sınıflandırma** problemlerinin vazgeçilmez loss fonksiyonudur. Özellikle görüntü sınıflandırma, spam tespiti, hastalık teşhisi gibi evet/hayır kararlarının verildiği durumlarda kullanılır. Matematiksel olarak, tahmin edilen olasılıkların log-likelihood değerini maksimize etmeye çalışır.

```python
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

### 🔍 Özellikleri:
- Sigmoid aktivasyon fonksiyonu ile birlikte kullanılır
- 0'a yakın yanlış tahminleri ve 1'e yakın doğru tahminleri teşvik eder
- Gradyan değerleri otomatik olarak ölçeklenir
- Dengesiz veri setlerinde class_weight parametresi ile desteklenebilir

## 3. Mean Squared Error (MSE)
**Regresyon** problemlerinin temel taşı olan MSE, tahmin edilen değerler ile gerçek değerler arasındaki farkın karesinin ortalamasını hesaplar. Bu fonksiyon, özellikle ev fiyatı tahmini, hava sıcaklığı öngörüsü gibi sürekli değer tahminlerinde tercih edilir.

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### 🔍 Özellikleri:
- Her zaman pozitif değer üretir, çünkü farkların karesi alınır
- Büyük hatalar karesel olarak cezalandırılır, bu nedenle aykırı değerlere karşı hassastır
- Türevi kolay hesaplanabilir, bu da gradyan inişi için avantaj sağlar
- Hedef değişken normal dağılıma sahip olduğunda optimal sonuç verir

## 4. Mean Absolute Error (MAE)
MSE'ye alternatif olan MAE, tahminler ile gerçek değerler arasındaki mutlak farkların ortalamasını alır. Aykırı değerlere karşı MSE'den daha az hassastır.

```python
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

### 🔍 Özellikleri:
- Mutlak değer kullanıldığı için aykırı değerlere karşı MSE'den daha dayanıklıdır
- Her zaman pozitif değer üretir
- Lineer ölçekleme sağlar, büyük hataları MSE kadar şiddetli cezalandırmaz
- Medyan tahminlerinde optimal sonuç verir

## 5. Huber Loss
MSE ve MAE'nin hibrit bir versiyonudur. Küçük hatalar için MSE gibi, büyük hatalar için MAE gibi davranır. Aykırı değerlere karşı MSE'den daha dirençlidir.

```python
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * np.abs(error) - 0.5 * np.square(delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))
```

### 🔍 Özellikleri:
- Delta parametresi ile MSE ve MAE arasında geçiş yapılabilir
- Aykırı değerlere karşı MSE'den daha dayanıklıdır
- Küçük hatalar için karesel, büyük hatalar için lineer ceza uygular
- Regresyon problemlerinde dengeli bir seçenek sunar

## 6. Focal Loss
Dengesiz veri setlerinde **sınıflandırma** problemleri için tasarlanmış özel bir kayıp fonksiyonudur. Zor örneklere daha fazla ağırlık vererek, modelin sık görülen örnekler yerine nadir görülen örneklere odaklanmasını sağlar.

```python
def focal_loss(y_true, y_pred, gamma=2.0):
    ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    pt = tf.exp(-ce)
    return tf.mean((1 - pt) ** gamma * ce)
```

### 🔍 Özellikleri:
- Dengesiz veri setlerinde performansı artırır
- Gamma parametresi ile kolay/zor örnekler arasındaki denge ayarlanabilir
- Binary Cross-Entropy'nin geliştirilmiş versiyonudur
- Nesne tespiti gibi sınıf dengesizliği olan problemlerde yaygın kullanılır

## 7. Kullback-Leibler Divergence (KL Divergence)
İki olasılık dağılımı arasındaki farkı ölçen bir kayıp fonksiyonudur. Özellikle Variational Autoencoder (VAE) ve probabilistik modellerde kullanılır.

```python
def kl_divergence(p, q):
    return tf.reduce_sum(p * tf.math.log(p / q))
```

### 🔍 Özellikleri:
- Dağılımlar arasındaki benzerliği ölçer
- Asimetrik bir metriktir (KL(P||Q) ≠ KL(Q||P))
- Information theory temelli bir yaklaşımdır
- VAE ve GAN gibi üretici modellerde sıkça kullanılır

## Loss Fonksiyonu Seçim Kriterleri

| Problem Tipi | Önerilen Loss Fonksiyonu | Kullanım Alanı | Özel Durumlar |
|--------------|-------------------------|----------------|----------------|
| Regresyon | MSE, MAE, Huber | Sayısal tahmin | Aykırı değer varsa MAE veya Huber tercih edilebilir |
| Binary Classification | Binary Cross-Entropy, Focal Loss | İkili sınıflandırma | Dengesiz veri setlerinde Focal Loss tercih edilebilir |
| Multi-class Classification | Categorical Cross-Entropy | Çoklu sınıflandırma | Sparse versiyonu bellek tasarrufu sağlar |
| Probabilistic Models | KL Divergence | Olasılık dağılımları | Autoencoder ve GAN'larda sıkça kullanılır |

### Loss Değişiminin Görselleştirmesi:

```
     Loss
     ^
     |
     |    .  Training Loss
     |      .
     |        .  Validation Loss
     |          .
     |            ..
     |              ...
     |                 ....
     +--------------------------> Epochs
```

## Önemli Notlar ve En İyi Uygulamalar:
- Loss değerinin mutlak büyüklüğü değil, eğitim boyunca gösterdiği eğilim önemlidir
- Validation loss'un training loss'tan belirgin şekilde yüksek olması overfitting göstergesidir
- Loss fonksiyonu seçimi, problemin doğası ve veri dağılımına uygun olmalıdır
- Özel durumlarda birden fazla loss fonksiyonu birleştirilebilir (custom loss)
- Loss değerlerinin ölçeklendirilmesi ve normalizasyonu model performansını etkileyebilir
- Early stopping için validation loss takip edilmelidir

## Kullanılan Tüm Kayıp Fonksiyonlarının Listesi

| Kullanım Alanı                    | Kayıp Fonksiyonu                            | Açıklama                                                                 |
|-----------------------------------|---------------------------------------------|--------------------------------------------------------------------------|
| **Sınıflandırma**                 | **Binary Crossentropy**                     | İkili sınıflandırma problemlerinde kullanılır.                          |
|                                   | **Categorical Crossentropy**                | One-hot encoded etiketlerle birlikte kullanılır.                         |
|                                   | **Sparse Categorical Crossentropy**        | Sınıf indeksleri ile etiketleme için kullanılır.                        |
|                                   | **Focal Loss**                              | Dengesiz veri setlerinde daha fazla odaklanmak için kullanılır.        |
|                                   | **Cohen's Kappa Loss**                      | Sınıflar arasındaki dengesizliği dikkate alır.                          |
|                                   | **Softmax Loss**                            | Çok sınıflı sınıflandırma için kullanılır.                              |
|                                   | **Contrastive Loss**                        | İki örnek arasındaki mesafeyi optimize eder.                           |
|                                   | **Triplet Loss**                            | Üçlü örnekler arasındaki mesafeyi optimize eder.                       |
| **Regresyon**                     | **Mean Squared Error (MSE)**                | Gerçek ve tahmin edilen değerler arasındaki ortalama kare farkını hesaplar. |
|                                   | **Mean Absolute Error (MAE)**               | Gerçek ve tahmin edilen değerler arasındaki ortalama mutlak farkı hesaplar. |
|                                   | **Huber Loss**                              | MSE ve MAE'nin bir kombinasyonu, hata küçükse MSE, büyükse MAE kullanır. |
|                                   | **Poisson Loss**                            | Sayım verileriyle çalışırken kullanılır.                                |
| **Görüntü Segmentasyonu**        | **Tversky Loss**                            | Yanlış pozitif ve negatif hataları ayırt eder.                         |
|                                   | **Jaccard Loss**                            | İki küme arasındaki benzerliği ölçer.                                  |
| **Olasılık Dağılımı Karşılaştırması** | **Kullback-Leibler Divergence**         | İki olasılık dağılımı arasındaki farkı ölçer.                          |
| **Özel Uygulamalar**             | **Custom Loss Functions**                   | Kullanıcılar tarafından belirli bir probleme göre özelleştirilmiş kayıplar. |
|                                   | **Adversarial Loss**                        | Üretici ve ayırt edici ağlar arasında denge kurar.                     |
|                                   | **Weighted Loss**                           | Belirli sınıflara daha fazla ağırlık vermek için tasarlanmıştır.       |

