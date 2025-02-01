# Matematik Temelleri

## 📍 Bölüm Haritası
- Önceki Bölüm: [Python ve Kütüphaneler](02-Python-Kutuphaneler.md)
- Sonraki Bölüm: [Perceptron ve Sinir Hücreleri](../02-Yapay-Sinir-Aglari/01-Perceptron.md)
- Tahmini Süre: 4-5 saat
- Zorluk Seviyesi: 🟢 Başlangıç

## 🎯 Hedefler
- Derin öğrenme için gerekli matematik kavramlarını anlama
- Temel lineer cebir ve kalkülüs bilgisi edinme
- Optimizasyon ve olasılık teorisi temellerini öğrenme
- Matematiksel notasyonu ve terminolojiyi kavrama

## 🎯 Öz Değerlendirme
- [ ] Temel lineer cebir işlemlerini yapabiliyorum
- [ ] Türev ve integral kavramlarını anlayabildim
- [ ] Optimizasyon problemlerini çözebiliyorum
- [ ] Olasılık ve istatistik hesaplamalarını yapabiliyorum

## 🚀 Mini Projeler
1. Gradyan İniş Simülasyonu
   - 2D ve 3D fonksiyonlar için gradyan inişi uygulayın
   - Farklı öğrenme oranlarının etkisini gözlemleyin
   - Sonuçları görselleştirin

2. Matris İşlemleri Uygulaması
   - Temel matris operasyonlarını içeren bir kütüphane yazın
   - Sinir ağı hesaplamalarını manuel olarak gerçekleştirin
   - Numpy sonuçlarıyla karşılaştırın

## 📑 Ön Koşullar
- Temel matematik bilgisi
- Fonksiyon kavramı
- Grafik okuma ve yorumlama
- Temel problem çözme becerisi

## 🔑 Temel Kavramlar
1. Lineer Cebir
2. Kalkülüs
3. Olasılık ve İstatistik
4. Optimizasyon
5. Matris İşlemleri

## Lineer Cebir
> 💡 İpucu: Matris işlemlerini iyi anlamak, derin öğrenme modellerinin çalışma prensibini kavramayı kolaylaştırır

### 1. Vektörler ve Matrisler
```python
import numpy as np

# Vektör oluşturma
v = np.array([1, 2, 3])

# Matris oluşturma
A = np.array([[1, 2], [3, 4]])
```

### 2. Türev
Türev, bir fonksiyonun anlık değişim oranıdır ve gradyan inişte kullanılır.

```python
# Otomatik türev hesaplama
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x * x  # y = x^2
dy_dx = tape.gradient(y, x)  # dy/dx = 2x
```

### 3. Kısmi Türevler
Çok değişkenli fonksiyonların her bir değişkene göre türevi.

```python
# Kısmi türevler
x = tf.Variable(2.0)
y = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = x*x + y*y  # z = x^2 + y^2
gradients = tape.gradient(z, [x, y])  # [∂z/∂x, ∂z/∂y]
```

### 4. Gradyan
Tüm kısmi türevlerin oluşturduğu vektör.

```python
def compute_gradients(model, x, y):
    with tf.GradientTape() as tape:
        loss = tf.keras.losses.mean_squared_error(y, model(x))
    return tape.gradient(loss, model.trainable_variables)
```

## Olasılık ve İstatistik

### 1. Temel Kavramlar
```python
# Ortalama ve standart sapma
data = tf.constant([1, 2, 2, 3, 4, 5, 5])
mean = tf.reduce_mean(data)
std = tf.math.reduce_std(data)

# Olasılık dağılımı
probs = tf.nn.softmax([1.0, 2.0, 3.0])  # Kategorik dağılım
```

### 2. Olasılık Dağılımları

#### Normal Dağılım
```python
# Normal dağılımdan örnekleme
samples = tf.random.normal(shape=[1000], mean=0.0, stddev=1.0)

# Histogram çizimi
import matplotlib.pyplot as plt
plt.hist(samples.numpy(), bins=30)
plt.title('Normal Dağılım')
plt.show()
```

## Optimizasyon

### 1. Gradyan İniş
```python
# Basit gradyan iniş
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def train_step(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.mean_squared_error(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### 2. Optimizasyon Algoritmaları
```python
# Adam optimizer
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)

# RMSprop optimizer
rmsprop = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9
)
```

## Matris İşlemleri

### 1. Determinant ve Ters Matris
```python
# 3x3 matris oluşturun
matrix = tf.random.normal([3, 3])

# Determinant hesaplama
det = tf.linalg.det(matrix)

# Ters matris
inv = tf.linalg.inv(matrix)
```

### 2. Türev ve İntegral
```python
# f(x) = x³ + 2x² - 4x + 1
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x**3 + 2*x**2 - 4*x + 1
dy_dx = tape.gradient(y, x)
```

## 📚 Önerilen Kaynaklar
- [Khan Academy Mathematics](https://www.khanacademy.org/math)
- [3Blue1Brown Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Deep Learning Book - Math Chapters](https://www.deeplearningbook.org/contents/part_basics.html)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. Temel matris işlemleri
2. Türev ve integral hesaplamaları

### Orta Seviye
1. Optimizasyon problemleri çözme
2. İstatistiksel analiz uygulamaları

### İleri Seviye
1. Gradyan hesaplamaları
2. Olasılık dağılımları analizi 