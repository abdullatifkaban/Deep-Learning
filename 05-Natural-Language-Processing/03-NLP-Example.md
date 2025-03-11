# Yelp Review İncelemesi - NLP Uygulaması

Bu uygulamada Yelp kullanıcı yorumlarını analiz eden bir derin öğrenme modeli geliştireceğiz. Model, yorumların içeriğine göre yıldız sayısını tahmin edecek.

> [!NOTE]
> Bu projede kullanılan verileri [buradan](../Data/yelp.csv) indirebilirsiniz.

## Veri Seti İncelemesi

```python
import pandas as pd

# Veri setini okuma
df = pd.read_csv('yelp.csv')

# İlk 5 satırı görüntüleme 
print(df.head())

# Sütun bilgilerini görüntüleme
print(df.columns)
```

## Metin Önişleme

```python
import re
import nltk
from nltk.corpus import stopwords

def preprocess_text(text):
    # Küçük harfe çevirme
    text = text.lower()
    
    # Özel karakterleri temizleme
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Stop words temizleme
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Yorumları önişleme
df['processed_text'] = df['text'].apply(preprocess_text)
```

**Anahtar Kavramlar:**
- **lowercase**: Tüm metni küçük harfe çevirerek standardizasyon sağlar
- **regex (re.sub)**: Özel karakterleri ve noktalama işaretlerini temizler
- **stopwords**: "the", "is", "at" gibi çok sık kullanılan anlamsız kelimeleri kaldırır
- **tokenization**: Metni kelimelere ayırma işlemi gerçekleştirir

## Model Oluşturma

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```
**Tokenizer oluşturma**
```python
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['processed_text'])
```

**Metinleri sayısal dizilere çevirme**

```python
X = tokenizer.texts_to_sequences(df['processed_text'])
X = pad_sequences(X, maxlen=200)
```

**Hedef değişkeni hazırlama**
```python
y = df['stars'] 
```

**Model mimarisi**
```python
model = Sequential()
model.add(Embedding(5000, 32, input_length=200))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

**Model Mimarisi Açıklaması:**

1. **Embedding Katmanı (5000, 32)**
    - Kelime vektörlerini 32 boyutlu yoğun vektörlere dönüştürür
    - Toplam 5000 benzersiz kelime için vektör oluşturur
    - Her kelime için özel bir sayısal temsil öğrenir

2. **İlk LSTM Katmanı (64 birim)**
    - Uzun-Kısa Süreli Bellek birimi
    - Metin içindeki uzun vadeli bağımlılıkları öğrenir
    - return_sequences=True ile çıktıyı bir sonraki LSTM'e aktarır
    - 64 boyutlu çıktı üretir

3. **İkinci LSTM Katmanı (32 birim)**
    - Daha kompakt bir temsil oluşturur
    - İlk LSTM'den gelen bilgiyi işler
    - 32 boyutlu final çıktı üretir

4. **Yoğun Katman (16 birim)**
    - ReLU aktivasyon fonksiyonu kullanır
    - Doğrusal olmayan özellikleri yakalar
    - Boyut azaltmaya devam eder

5. **Son Yoğun Katman (1 birim)**
    - Linear aktivasyon ile yıldız sayısını tahmin eder
    - Sürekli bir değer çıktısı üretir (1-5 arası)

Model, ortalama kare hata (MSE) kaybı ve ortalama mutlak hata (MAE) metriği ile derlenir. Adam optimizer kullanılarak eğitilir.


## Model Eğitimi

```python
# Modeli eğitme
history = model.fit(X, y, epochs=10, batch_size=32, 
                    validation_split=0.2, verbose=1)
```

## Model Başarısı ve Testi

```python
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
```

**Eğitim geçmişini görselleştirme**
```python
plt.figure(figsize=(10,6))
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
```

**Tahminler**
```python
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² Skoru: {r2:.3f}")
```

```python
test_text = ["This restaurant was amazing! Great food and service."]
test_processed = [preprocess_text(text) for text in test_text]
test_seq = tokenizer.texts_to_sequences(test_processed)
test_pad = pad_sequences(test_seq, maxlen=200)
prediction = model.predict(test_pad)
print(f"Tahmini Yıldız Sayısı: {prediction[0][0]:.1f}")
```

Model başarısını değerlendirmek için hem MAE hem de R² metriklerini kullandık. R² skoru modelin hedef değişkendeki varyasyonu ne kadar iyi açıkladığını gösterir. 1'e yakın R² değerleri daha iyi performansa işaret eder.