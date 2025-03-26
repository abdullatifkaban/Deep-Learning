# Konvolüsyonel Sinir Ağları (CNN)

## Giriş

Konvolüsyonel Sinir Ağları (CNN), özellikle görüntü işleme ve bilgisayarla görme alanlarında yaygın olarak kullanılan bir derin öğrenme mimarisidir. CNN'ler, görüntülerdeki uzamsal hiyerarşiyi öğrenme yetenekleri sayesinde yüksek doğruluk oranlarına ulaşır.

## CNN Katmanları

### 1. Konvolüsyon Katmanı

Konvolüsyon katmanı, görüntüden özellikler çıkarmak için filtreler (kernels) kullanır. Her filtre, görüntü üzerinde kaydırılarak (convolve) özellik haritaları oluşturur. Bu katman, görüntüdeki kenarlar, köşeler ve dokular gibi düşük seviyeli özellikleri öğrenir.

```python
from tensorflow.keras.layers import Conv2D

# 32 filtreli, 3x3 boyutunda konvolüsyon katmanı
conv_layer = Conv2D(32, (3, 3), activation='relu')
```

Konvolüsyon işlemi, her bir filtreyi görüntü üzerinde kaydırarak piksel değerleri ile filtre değerlerinin çarpımını alır ve bu değerleri toplar. Bu işlem, görüntünün belirli özelliklerini vurgulayan bir özellik haritası oluşturur.

### 2. Havuzlama (Pooling) Katmanı

Havuzlama katmanı, özellik haritalarının boyutunu azaltarak hesaplama maliyetini düşürür ve modelin aşırı öğrenmesini (overfitting) önler. En yaygın kullanılan havuzlama türü max pooling'dir. Max pooling, belirli bir alandaki en yüksek piksel değerini seçer ve bu değeri yeni bir özellik haritasına yerleştirir.

```python
from tensorflow.keras.layers import MaxPooling2D

# 2x2 boyutunda max pooling katmanı
pooling_layer = MaxPooling2D(pool_size=(2, 2))
```

Havuzlama işlemi, modelin konum değişikliklerine karşı daha dayanıklı olmasını sağlar ve özellik haritalarının boyutunu azaltarak hesaplama maliyetini düşürür.

### 3. Tam Bağlantılı (Fully Connected) Katman

Tam bağlantılı katman, tüm nöronların birbirine bağlı olduğu bir katmandır. Genellikle sınıflandırma görevlerinde kullanılır. Bu katman, konvolüsyon ve havuzlama katmanlarından gelen özellikleri alır ve bu özellikleri kullanarak sınıflandırma yapar.

```python
from tensorflow.keras.layers import Dense

# 128 nöronlu tam bağlantılı katman
fc_layer = Dense(128, activation='relu')
```

Tam bağlantılı katman, görüntüdeki yüksek seviyeli özellikleri öğrenir ve bu özellikleri kullanarak sınıflandırma yapar. Bu katman, genellikle softmax aktivasyon fonksiyonu ile birlikte kullanılır ve bu sayede sınıflandırma sonuçları olasılık değerleri olarak elde edilir.

## Sonuç

Konvolüsyonel Sinir Ağları (CNN), görüntü işleme ve bilgisayarla görme alanlarında güçlü bir araçtır. Konvolüsyon, havuzlama ve tam bağlantılı katmanlar, CNN'lerin temel yapı taşlarıdır ve bu katmanlar sayesinde görüntülerdeki özellikler öğrenilir ve sınıflandırma yapılır.
