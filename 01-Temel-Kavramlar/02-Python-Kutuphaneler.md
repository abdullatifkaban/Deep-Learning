# Python ve Gerekli Kütüphaneler

## 🎯 Hedefler
- Python programlama temellerini öğrenme
- Veri bilimi kütüphanelerini tanıma
- Derin öğrenme framework'lerini anlama
- Temel veri işleme becerilerini geliştirme

## 📑 Ön Koşullar
- Temel programlama mantığı
- Terminal/komut satırı kullanımı
- Pip paket yöneticisi bilgisi
- Temel matematik bilgisi

## 🔑 Temel Kavramlar
1. Python Temelleri
2. NumPy ve Pandas
3. Matplotlib ve Seaborn
4. TensorFlow ve PyTorch
5. Scikit-learn

## Python Temelleri
> Zorluk Seviyesi: 🟢 Başlangıç

> 💡 İpucu: Python'un temel veri yapılarını ve fonksiyonlarını iyi öğrenmek, ilerideki konuları anlamayı kolaylaştırır

### 1. Veri Yapıları
```python
# Liste
numbers = [1, 2, 3, 4, 5]

# Sözlük
person = {'name': 'Ali', 'age': 25}

# Tuple
coordinates = (40.7128, -74.0060)
```

### 2. Kontrol Yapıları
```python
# if-elif-else
x = 5
if x > 0:
    print("Pozitif")
elif x < 0:
    print("Negatif")
else:
    print("Sıfır")

# for döngüsü
for i in range(5):
    print(i)

# while döngüsü
sayac = 0
while sayac < 5:
    print(sayac)
    sayac += 1
```

## NumPy
NumPy, bilimsel hesaplamalar için temel kütüphanedir.

```python
import numpy as np

# Array oluşturma
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Özel arrayler
zeros = np.zeros((3, 3))    # Sıfır matrisi
ones = np.ones((2, 2))      # Birler matrisi
random = np.random.rand(3,3) # Rastgele sayılar
```

## TensorFlow
TensorFlow, derin öğrenme için modern bir kütüphanedir.

### Temel Tensor İşlemleri
```python
import tensorflow as tf

# Tensor oluşturma
x = tf.constant([1., 2., 3.])
y = tf.zeros([2, 3])
z = tf.random.normal([3, 3])

# GPU kullanımı
if tf.config.list_physical_devices('GPU'):
    print("GPU bulundu!")

# Otomatik türev
with tf.GradientTape() as tape:
    x = tf.Variable(3.0)
    y = x * x
grad = tape.gradient(y, x)  # dy/dx = 2x = 6.0
```

### Katmanlar ve Model Oluşturma
```python
# Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Functional API
inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
```

## Pandas
Pandas, veri analizi ve manipülasyonu için kullanılır.

```python
import pandas as pd

# DataFrame oluşturma
data = {
    'isim': ['Ali', 'Veli', 'Ayşe'],
    'yaş': [25, 30, 35],
    'şehir': ['İstanbul', 'Ankara', 'İzmir']
}
df = pd.DataFrame(data)

# Veri okuma
# df = pd.read_csv('veri.csv')
# df = pd.read_excel('veri.xlsx')
```

## Matplotlib
Matplotlib, veri görselleştirme kütüphanesidir.

```python
import matplotlib.pyplot as plt

# Temel grafik
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Çizgi Grafik')
plt.xlabel('X ekseni')
plt.ylabel('Y ekseni')
plt.show()

# TensorFlow eğitim geçmişi görselleştirme
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()
```

## Scikit-learn
Scikit-learn, makine öğrenmesi için kullanılan bir kütüphanedir.

### Model Eğitimi
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Veri setini böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Veri ön işleme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TensorFlow modeli eğitimi
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## 📚 Önerilen Kaynaklar
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [NumPy Documentation](https://numpy.org/doc/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. Temel Python programlama
2. NumPy array işlemleri

### Orta Seviye
1. Pandas ile veri analizi
2. Matplotlib ile görselleştirme

### İleri Seviye
1. TensorFlow/PyTorch projeleri
2. Kütüphane entegrasyonları 