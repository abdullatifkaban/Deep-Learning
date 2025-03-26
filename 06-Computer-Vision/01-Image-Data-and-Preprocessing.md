# Görüntü Verisi ve Ön İşleme

## Görüntü Verisinin Özellikleri

Görüntüler, JPEG, PNG, BMP gibi çeşitli formatlarda temsil edilebilen bir veri türüdür. Her görüntü piksellerden oluşur ve her piksel, yoğunluğunu veya rengini temsil eden bir değere sahiptir. Görüntü verisinin temel özellikleri şunlardır:

- **Çözünürlük**: Görüntüdeki piksel sayısı, genellikle genişlik x yükseklik olarak temsil edilir (örneğin, 1920x1080).
- **Renk Derinliği**: Tek bir pikselin rengini temsil etmek için kullanılan bit sayısı. Yaygın renk derinlikleri 8-bit (256 renk), 16-bit ve 24-bit (gerçek renk) olarak bilinir.
- **Kanallar**: Görüntüler birden fazla kanala sahip olabilir. Örneğin, RGB görüntüleri üç kanala sahiptir (Kırmızı, Yeşil, Mavi), gri tonlamalı görüntüler ise yalnızca bir kanala sahiptir.

## Ön İşleme Teknikleri

Ön işleme, görüntülerin kalitesini artırmak ve daha ileri analizler için hazırlamak amacıyla bilgisayarla görmede önemli bir adımdır. Yaygın ön işleme teknikleri şunlardır:

### 1. Yeniden Boyutlandırma

Yeniden boyutlandırma, görüntünün boyutlarını değiştirir. Genellikle sinir ağları için giriş boyutunu standartlaştırmak amacıyla kullanılır.

```python
from PIL import Image

image = Image.open('example.jpg')
resized_image = image.resize((224, 224))
resized_image.show()
```

### 2. Normalizasyon

Normalizasyon, piksel değerlerini belirli bir aralığa ölçeklendirir, genellikle [0, 1] veya [-1, 1]. Bu, eğitim sırasında daha hızlı yakınsamaya yardımcı olur.

```python
import numpy as np

image_array = np.array(image)
normalized_image = image_array / 255.0
```

### 3. Gri Tonlamaya Dönüştürme

Bir görüntüyü gri tonlamaya dönüştürmek, renk bilgisini kaldırarak karmaşıklığı azaltır, bu da belirli uygulamalar için yararlı olabilir.

```python
grayscale_image = image.convert('L')
grayscale_image.show()
```

### 4. Veri Artırma

Dönme, çevirme ve kırpma gibi veri artırma teknikleri, eğitim veri setinin boyutunu yapay olarak artırmak ve modelin dayanıklılığını artırmak için kullanılır.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Tek bir görüntüye artırma uygulama örneği
augmented_image = datagen.random_transform(image_array)
```

### 5. Histogram Eşitleme

Histogram eşitleme, en sık görülen yoğunluk değerlerini yayarak bir görüntünün kontrastını artırır.

```python
import cv2

image_cv = cv2.imread('example.jpg', 0)  # Görüntüyü gri tonlamada yükle
equalized_image = cv2.equalizeHist(image_cv)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# Sonuç

Ön işleme, görüntü verilerini bilgisayarla görme görevleri için hazırlamada önemli bir adımdır. Bu teknikleri uygulayarak, görüntülerin kalitesini artırabilir ve modellerimizin performansını iyileştirebiliriz.
