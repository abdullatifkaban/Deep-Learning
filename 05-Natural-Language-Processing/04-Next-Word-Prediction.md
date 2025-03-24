# Sonraki Kelime Tahmin Modeli

Bu modelde Ömer Seyfettin hikayeleri kullanılarak derin öğrenme modeli eğitilecek ve metin üretmeye çalışılacaktır.

> [!NOTE]
> Bu projede kullanılan verileri [buradan](../Data/hikayeler.txt) indirebilirsiniz.

## Verileri Okuyalım

Gerekli kütüphaneleri `import` ettikten sonra metin dosyasından Ömer Seyfettin'e ait seçme hikayeleri okuyarak `text` değişkenine aktaralım.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Read the text file
with open('hikayeler.txt', 'r', encoding='utf-8') as file:
    text = file.read()
```
## Metin Önişleme

Öncelikle metinde yer alan paragraf başı ve satır sonu işaretlerini temizleyelim.

```python
import re
text = re.sub(r'[\n\r]+', ' ', text)
```

Metni `tokenize` ederek cümle listesi haline getirelim.

```python
from nltk.tokenize import word_tokenize, sent_tokenize
cumleler = sent_tokenize(text)
```

Her bir cümle içindeki gereksiz karakterleri temizleyelim.

```py
def clean_sentence(sentence):
    # Özel karakterleri ve sayıları kaldır, Türkçe harfleri koru
    sentence = re.sub(r'[^a-zA-ZçğİıöşüÇĞİİÖŞÜâîû\s.!?]', '', sentence)
    # Küçük harfe çevir
    sentence = sentence.lower()
    # Boşlukları temizle
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

temiz = [clean_sentence(cumle) for cumle in cumleler]
```

Herbir kelimeye bir sayı atayarak kelime havuzu (`tokenizer`) oluşturup toplam kelime sayısını bulalım. 
```py
tokenizer = Tokenizer()
tokenizer.fit_on_texts(temiz)
toplam_ks = len(tokenizer.word_index) + 1
```
Yukarıda oluşturduğumuz `tokenizer`i kullanarak `temiz`lenmiş her cümle için kelime dizilerini oluşturup ve her diziyi kısmi olarak kesip yeni bir listeye ekleyelim. Yani, her cümlenin kelimelerinin farklı kombinasyonlarını içeren sayısal dizilerden oluşan bir liste (`input_sequences`) elde edelim.
```py
input_sequences = [tokenizer.texts_to_sequences([line])[0][:i+1] 
    for line in temiz
    for i in range(1, len(tokenizer.texts_to_sequences([line])[0]))]
```
En uzun cümlenin kelime sayısını (`max_sequence_len`) hesaplayalım.
```py
max_sequence_len = max([len(seq) for seq in input_sequences])
```

En uzun cümlenin kelime sayısına (`max_sequence_len`) göre oluşturduğumuz sayısal dizilerin (`input_sequences`) boyutlarını eşitleyelim.

```py
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
```

## Model Oluşturma

Modelin girdi ve hedef değişkenlerini tanımlayalım. Oluşturduğumuz sayısal dizilerin (`input_sequences`) son elemanlarını hedef alan olarak belirleyip geriye kalanları da girdi alanları olarak tanımlıyoruz.
```py
x = input_sequences[:, :-1]
y = input_sequences[:, -1]
```
Hedef değişkeni `y` kategori değişkeni haline getirelim. Yani `one-hot encoding` yaparak toplam kelime sayısı kadar sütuna dönüştürüyoruz.
```py
y = np.array(tf.keras.utils.to_categorical(y, num_classes=toplam_ks))
```
Sinir ağımızın modelini tanımlayalım.
```py
model = Sequential()
model.add(Embedding(toplam_ks, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(toplam_ks, activation='softmax'))
```


- **model = Sequential()**: Bir Sequential model oluşturur; bu, katmanların sıralı olarak ekleneceği basit bir model yapısıdır.

- **model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))**: Girdi verilerini sayısal kelime gömme (embedding) katmanına dönüştürür.
    * `total_words`: Kelime sayısını belirtir.
    * `100`: Her kelimenin 100 boyutlu bir vektörle temsil edileceğini belirtir.
    * `input_length`: Giriş dizisinin uzunluğudur (maksimum dizinin uzunluğundan 1 eksik).

- **model.add(LSTM(150, return_sequences=True))**: İlk LSTM (Uzun Kısa Süreli Bellek) katmanını ekler.
    * `150`: LSTM hücrelerinin sayısını belirtir.
    * `return_sequences`=True: Sonuçları her zaman adımda döndürür, böylece sonraki LSTM katmanı için bir dizi çıktı sağlar.

- **model.add(LSTM(150))**: İkinci bir LSTM katmanı ekler.
    Bu katman, önceki katmanın çıktısını alır, ancak return_sequences parametresi varsayılan olarak False olduğu için yalnızca son çıktıyı döndürür.

- **model.add(Dense(total_words, activation='softmax'))**: Çıkış katmanını ekler.
    * `Dense`: Tam bağlantılı bir katmandır.
    * `total_words`: Çıkış biriminin sayısını belirtir (sınıf sayısı).
    * `activation`='softmax': Çıkışların olasılık dağılımına dönüştürülmesini sağlar, böylece her sınıf için bir olasılık değeri elde edilir.

Modeli derleyelim.

```py
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Modeli eğitelim.
```py
model.fit(x, y, epochs=100, verbose=1)
```

## Modeli Kaydetme

Modeli bir uygulamada kullanabilmek için tüm özellikleri ile kaydetmek gerekir.

```py
# Sinir ağı modeli
model.save('omer_model.h5')

# Kelime ve numara havuzu
import pickle
pickle.dump(tokenizer, open("omer_tokenizer.pkl", "wb"))

# Kelime-index sözlüğünün kaydedilmesi
import pandas as pd
word_index_df = pd.DataFrame(list(tokenizer.word_index.items()), columns=['word', 'index'])
# Sözlükte rakam istemiyorsanız aşağıdaki satırı kullanabilirsiniz.
word_index_df = word_index_df[~word_index_df["word"].str.isnumeric()]
word_index_df.to_csv('omer_word_index.csv', index=False)
word_index_df.info()
```

## Metin Üretme

Sonraki kelimeyi tahmin eden bir fonksiyon tanımlayalım.
```py
import random

word_index_df = word_index_df[~word_index_df["word"].str.isnumeric()]

def pre_words(seed_text, n):
    for _ in range(n):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text
```

Fonksiyonu kullanarak `Kaşağı` kelimesinden sonra gelecek `20` kelimeyi tahmin edelim.

```py
pre_words("Kaşağı", 20)
```

Verilen bir cümleyi tamamlayacak başka bir fonksiyon tanımlayalım
```py
def generate_text(seed_text, next_words, model, max_sequence_length, temperature=1.0):
    generated = seed_text
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([generated])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        
        # Apply temperature sampling
        predicted = np.log(predicted + 1e-7) / temperature
        exp_predicted = np.exp(predicted)  
        predicted = exp_predicted / np.sum(exp_predicted)  
        
        predicted_word_index = np.random.choice(range(len(predicted)), p=predicted)

        # Get the word from the index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
                
        generated += " " + output_word        
    
    return generated
```

Fonksiyonu kullanarak `kızılelma neresi` ifadesini takip edecek `50` kelimelik metin üretelim.

```py
seed_text = "kızılelma neresi"
generated_text = generate_text(seed_text, next_words=50, model=model, max_sequence_length=max_sequence_len, temperature=0.8)
generated_text
```

# Sonuç
Ömer Seyfettinin seçme hikayelerinden elde edilen metin kullanılarak eğitilen bir sinir ağı ile sonraki kelimeyi tahmin edecek bir model geliştirdik. Bu model kullanılarak Ömer Seyfettin benzeri hikayeler yazmak mümkündür. Modelin başarısını artırmak için daha çok metin ve daha derin ağlarla eğitim yapılabilir. Metin üretme aşamasında ise cümlenin öğeleri `Part of Speech (POS)` tespit edilerek daha başarılı cümle yapıları oluşturulabilir.

Oluşturulan model ile ilgili dosyaları aşağıdaki linklerden indirebilirsiniz:
* [Yapay Sinir Ağı Modeli](../Data/omer_model.h5)
* [Tokenizer](../Data/omer_tokenizer.pkl)
* [Kelime Havuzu](../Data/omer_word_index.csv)