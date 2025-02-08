# Derin Öğrenme için Python Kütüphaneleri

## İçerik
1. [Gerekli Kütüphaneler](#gerekli-kütüphaneler)
2. [Kurulum Rehberi](#kurulum-rehberi)
3. [Temel Kullanım Örnekleri](#temel-kullanım-örnekleri)

## Gerekli Kütüphaneler

### 1. NumPy
NumPy, bilimsel hesaplamalar için temel kütüphanedir. Çok boyutlu diziler ve matrisler üzerinde hızlı işlemler yapmanızı sağlar.

```python
import numpy as np

# Örnek NumPy işlemleri
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Dizi şekli: {arr.shape}")  # (2, 3)
print(f"Boyut sayısı: {arr.ndim}") # 2
print(f"Toplam: {arr.sum()}")      # 21
```

### 2. Pandas
Pandas, veri manipülasyonu ve analizi için kullanılan güçlü bir kütüphanedir.

```python
import pandas as pd

# Örnek veri çerçevesi oluşturma
data = {
    'isim': ['Ali', 'Ayşe', 'Mehmet'],
    'yaş': [25, 30, 35],
    'şehir': ['İstanbul', 'Ankara', 'İzmir']
}
df = pd.DataFrame(data)
print(df.head())
```

### 3. Matplotlib
Veri görselleştirme için kullanılan temel kütüphanedir.

```python
import matplotlib.pyplot as plt
import numpy as np

# Basit bir grafik çizimi
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sinüs Dalgası')
plt.show()
```

### 4. TensorFlow
Google tarafından geliştirilen açık kaynaklı derin öğrenme framework'üdür.

```python
import tensorflow as tf

# Basit bir sinir ağı katmanı oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Kurulum Rehberi

Gerekli kütüphaneleri pip kullanarak kurabilirsiniz:

```bash
# Sanal ortam oluşturma (önerilen)
python -m venv deeplearning_env
source deeplearning_env/bin/activate  # Linux/Mac
# veya
.\deeplearning_env\Scripts\activate    # Windows

# Kütüphanelerin kurulumu
pip install numpy pandas matplotlib
pip install tensorflow  # CPU versiyonu
```

GPU desteği için:
- TensorFlow: `pip install tensorflow-gpu`

## Temel Kullanım Örnekleri

### Veri Ön İşleme Örneği

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Örnek veri oluşturma
data = np.random.randn(100, 3)
df = pd.DataFrame(data, columns=['özellik1', 'özellik2', 'özellik3'])

# Veri standardizasyonu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print("Ölçeklendirilmiş veri şekli:", scaled_data.shape)
```

### Basit Bir Sinir Ağı Eğitimi (TensorFlow ile)

```python
import tensorflow as tf

# MNIST veri setini yükleme
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Veriyi ön işleme
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Model derleme
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Eğitim
model.fit(x_train, y_train, epochs=5)
```

## İpuçları ve En İyi Uygulamalar

1. Her proje için ayrı bir sanal ortam kullanın
2. Kütüphane sürümlerini `requirements.txt` dosyasında belgelendirin
3. GPU kullanıyorsanız, donanımınıza uygun CUDA sürümünü kullandığınızdan emin olun
4. Büyük veri setleriyle çalışırken bellek yönetimine dikkat edin
5. Düzenli olarak kütüphaneleri güncelleyin, ancak projeniz kararlı çalışıyorsa sürüm değişikliklerinde dikkatli olun

> [!TIP]
> Nvidia GPU kullanıyorsanız, tensorflow kurulumu için [buraya](https://github.com/abdullatifkaban/Tensorflow) göz atabilirsiniz.

## 📚 Önerilen Kaynaklar
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [NumPy Documentation](https://numpy.org/doc/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) 
