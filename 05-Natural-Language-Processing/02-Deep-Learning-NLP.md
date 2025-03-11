# Derin Öğrenmede NLP Modelleri

## 1. Giriş
Doğal Dil İşleme (NLP) alanında derin öğrenme modelleri, metin verilerini analiz etme ve anlama konusunda harika sonuçlar üretmiştir. Son yıllarda, özellikle Transformer mimarisi gibi gelişmeler sayesinde, NLP alanında çığır açıcı ilerlemeler kaydedilmiştir.

Doğal Dil İşleme (NLP), günümüz teknoloji dünyasında kritik bir role sahiptir. **İnsan ve makine arasındaki iletişimi geliştirerek** daha doğal ve etkili etkileşimler sağlar. Özellikle büyük metin verilerinden **anlamlı bilgiler çıkarma** konusunda güçlü yeteneklere sahiptir. Otomatik **dil çevirisi** ve **metin özetleme** gibi pratik uygulamalarla günlük hayatı kolaylaştırırken, **duygu analizi** ve **metin sınıflandırma** gibi ileri düzey analizlerle de iş süreçlerine değer katar. Bu teknolojiler, hem bireysel kullanıcıların hem de kurumsal sistemlerin verimliliğini artırarak, modern dijital dönüşümün temel yapı taşlarından birini oluşturur.

Bu teknoloji, insan **dilinin karmaşık yapılarını anlama** ve işleme konusunda olağanüstü yetenekler sergilemektedir. Geleneksel yöntemlerin aksine, derin öğrenme modelleri veriyi ham formundan çıktıya kadar **uçtan uca (end-to-end) işleyebilmektedir**. Özellikle **bağlam tabanlı kelime temsilleri** sayesinde, kelimelerin farklı anlamlarını ve kullanımlarını başarıyla yakalayabilmektedir. Bu gelişmiş özellikler, çeşitli NLP görevlerinde **yüksek doğrulukta sonuçlar** elde edilmesini sağlamıştır.


## 2. Temel NLP Modelleri

### 2.1 Word Embeddings (Kelime Gömme)
Word Embeddings (Kelime Gömme), kelimeleri bilgisayarların anlayabileceği sayısal değerlere dönüştürme yöntemidir. Bunu günlük hayattan bir örnekle açıklayalım:

**🗺️ Benzetme:** Bir şehir haritasında her lokasyonun koordinatlarla (x,y) temsil edilmesi gibi, Word Embeddings de her kelimeyi çok boyutlu bir uzayda koordinatlarla temsil eder.

**Temel Özellikleri:**
* Her kelime bir vektör (sayı dizisi) ile temsil edilir
* Benzer anlamlı kelimeler, vektör uzayında birbirine yakın konumlanır
* Kelimeler arasındaki ilişkiler matematiksel olarak hesaplanabilir

**Örnek:**

```python
# Word Embedding örneği
kral = [0.2, 0.5, -0.3, 0.8]
kraliçe = [0.1, 0.6, -0.2, 0.9]
adam = [-0.4, 0.3, 0.2, 0.1]
kadın = [-0.3, 0.4, 0.3, 0.2]
```

Bu örnekte her kelime 4 boyutlu bir vektörle temsil edilmiştir. Benzer kelimeler (kral-kraliçe veya adam-kadın) birbirine yakın değerler alır.

```python
# Word Embeddings örneği
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Örnek metin
text = ["derin öğrenme harika", "nlp çok güzel"]

# One-hot encoding
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)

# Vektör gösterimi
print("Kelime vektörleri:")
print(X.toarray())
print("\nKelime listesi:", vectorizer.get_feature_names_out())
```
#### Word2Vec
Word2Vec, kelime gömme öğrenmek için kullanılan popüler bir tekniktir. İki farklı model mimarisi (CBOW ve Skip-gram) kullanarak kelimelerin vektör temsillerini öğrenir. 

- CBOW (Continuous Bag of Words): CBOW modeli, bağlam kelimelerden hedef kelimeyi tahmin eder
- Skip-gram modeli: CBOW modelinin tersini yapar
- Negative Sampling: Eğitim sürecini hızlandırmak için kullanılan etkili bir optimizasyon tekniğidir

```python
from gensim.models import Word2Vec

# Örnek veri
sentences = [["ben", "nlp", "öğreniyorum"], ["derin", "öğrenme", "harika"]]

# Word2Vec modeli
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Benzer kelimeleri bulma
similar_words = model.wv.most_similar('nlp')
```

#### GloVe (Global Vectors)
GloVe, kelime vektörlerini oluşturmak için kelime eş-oluşum istatistiklerini kullanan güçlü bir kelime gömme yöntemidir. Word2Vec'ten farklı olarak, global matris faktörizasyonu yaklaşımını kullanarak tüm korpustaki kelime ilişkilerini bir arada değerlendirir. Bu sayede kelimelerin semantik ve sözdizimsel özelliklerini daha iyi yakalayabilir.

- Global matris faktörizasyonu
- Eş-oluşum olasılıkları

```python
from torchtext.vocab import GloVe

# GloVe vektörlerini yükleme
glove = GloVe(name='6B', dim=100)
```

#### FastText
FastText, Facebook tarafından geliştirilen ve Word2Vec'in geliştirilmiş bir versiyonu olan kelime gömme yöntemidir. Kelimeleri karakter n-gramlarına bölerek işler, bu sayede bilinmeyen veya nadir kelimelerin vektör temsillerini de oluşturabilir. Bu özelliği sayesinde, özellikle morfolojik açıdan zengin dillerde etkili sonuçlar üretir.

- Karakter n-gram'ları
- OOV (Out-of-Vocabulary) kelime desteği

```python
from gensim.models import FastText

# FastText modeli
model = FastText(sentences, vector_size=100, window=5, min_count=1)
```

#### Uygulama Alanları
- Semantik benzerlik analizi
- Metin sınıflandırma
- Makine çevirisi
- Duygu analizi

### 2.2 Recurrent Neural Networks (RNN)
Tekrarlayan Sinir Ağları (RNN), özellikle metin, ses ve zaman serisi gibi sıralı verileri işlemek için tasarlanmış özel bir yapay sinir ağı mimarisidir. Geleneksel sinir ağlarından farklı olarak, RNN'ler önceki adımlardaki bilgileri hafızada tutarak, mevcut girdiyi işlerken geçmiş bilgileri de kullanabilir.

**Temel Özellikleri:**
* Sıralı veri işleme yeteneği
* Dahili hafıza mekanizması
* Her adımda aynı parametreleri kullanma
* Değişken uzunluktaki girdileri işleyebilme

#### RNN'lerin Temel Yapısı
- Giriş Katmanı (Input Layer)
- Gizli Katman (Hidden Layer)
- Çıkış Katmanı (Output Layer)
- Geri Besleme Bağlantıları (Feedback Connections)

#### RNN Türleri
- Tek Yönlü RNN (Unidirectional RNN)
- Çift Yönlü RNN (Bidirectional RNN)
- Çok Katmanlı RNN (Multi-layer RNN)

```python
import tensorflow as tf

# Basit RNN modeli
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.SimpleRNN(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Çift Yönlü RNN örneği
bidirectional_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(128)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
#### RNN'lerin Avantajları
- Değişken uzunluktaki dizileri işleyebilme
- Önceki bilgileri hafızada tutabilme
- Sıralı verilerde başarılı sonuçlar

#### RNN'lerin Zorlukları
- Vanishing Gradient Problemi
- Uzun Vadeli Bağımlılıkları Öğrenememe
- Yavaş Eğitim Süreci


#### RNN Uygulama Alanları
- Metin Sınıflandırma
- Dil Modelleme
- Konuşma Tanıma
- Makine Çevirisi
- Zaman Serisi Analizi

## 3. Modern NLP Mimarileri

### 3.1 LSTM ve GRU
LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit), RNN'lerin vanishing gradient problemini çözmek için geliştirilmiş özel mimari yapılardır.

**Temel Avantajları:**
* Uzun vadeli bağımlılıkları öğrenebilme
* Vanishing gradient problemini çözme
* Seçici bilgi akışı kontrolü
* Adaptif hafıza mekanizması

**🔄 Benzetme:** LSTM ve GRU'yu bir güvenlik kontrol noktasına benzetebiliriz:
- LSTM: Dört farklı kontrol noktası (gate) ile detaylı güvenlik kontrolü
- GRU: İki kontrol noktası ile daha basit ama etkili kontrol

#### Temel Yapıları:
LSTM:
```
           ┌─────────┐
           │  Forget │
Input ──►  │   Input │  ──► Output
           │ Output  │
           │   Cell  │
           └─────────┘
```

GRU:
```
           ┌─────────┐
Input ──►  │ Update  │  ──► Output
           │  Reset  │
           └─────────┘
```



#### LSTM Mimarisi
- Forget Gate: Gereksiz bilgileri unutmayı sağlar
- Input Gate: Yeni bilgilerin hücreye girişini kontrol eder
- Output Gate: Hücre çıktısını düzenler
- Cell State: Uzun vadeli bilgileri saklar

```python
# LSTM modeli örneği
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(10)
])
```

#### GRU Mimarisi
- Reset Gate: Önceki bilgileri sıfırlamayı kontrol eder
- Update Gate: Yeni bilgilerin güncellenmesini yönetir

```python
# GRU modeli örneği
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(10)
])
```

#### LSTM ve GRU Karşılaştırması
- GRU daha basit yapı ve daha hızlı eğitim
- LSTM daha karmaşık ve güçlü hafıza mekanizması
- GRU daha az parametre ve bellek kullanımı

#### Uygulama Alanları
- Dizi-Dizi Dönüşümleri
- Zaman Serisi Tahmini
- Metin Oluşturma
- Müzik Kompozisyonu

#### Başarılı LSTM/GRU Uygulamaları
- Dil Modellemesi
- Konuşma Tanıma
- Makine Çevirisi
- El Yazısı Tanıma

### 3.2 Transformer Mimarisi
Transformer mimarisi, 2017 yılında tanıtılan ve NLP alanında devrim yaratan yeni bir yapay sinir ağı modelidir. Bu mimari, önceki modellerin aksine RNN veya LSTM gibi tekrarlayan yapılar kullanmaz. Bunun yerine, "attention" (dikkat) mekanizması ile çalışır.

**🎯 Benzetme:** Transformer'ı bir sınıftaki öğrenci-öğretmen ilişkisine benzetebiliriz:
- Öğretmen (attention mekanizması) tüm öğrencilere (kelimelere) aynı anda bakabilir
- Önemli gördüğü noktalara daha fazla dikkat eder
- Her öğrenciyi diğer öğrencilerle ilişkilendirerek değerlendirir

#### Ana Bileşenler

1. **Multi-head Attention (Çoklu Başlı Dikkat)**
    - Birden fazla dikkat mekanizması paralel çalışır
    - Her biri farklı özelliklere odaklanır
    - Tıpkı bir konuyu farklı açılardan incelemek gibi

2. **Positional Encoding (Konum Kodlama)**
    - Kelimelerin cümle içindeki sırasını kodlar
    - Matematiksel sinüs ve kosinüs fonksiyonları kullanır
    - Böylece model, kelimelerin sırasını anlayabilir

3. **Feed-Forward Networks (İleri Beslemeli Ağlar)**
    - Basit ama güçlü sinir ağı katmanları
    - ReLU aktivasyon fonksiyonu kullanır
    - Son işlemleri gerçekleştirir


#### Transformerin Çalışma Prensibi

1. **Giriş Aşaması:**
    - Metin önce kelimelere ayrılır
    - Her kelime sayısal vektörlere dönüştürülür
    - Konum bilgisi eklenir

2. **İşleme Aşaması:**
    - Attention mekanizması devreye girer
    - Her kelime diğer kelimelerle ilişkilendirilir
    - Önemli bağlantılar vurgulanır

3. **Çıkış Aşaması:**
    - Son işlemler yapılır
    - Sonuçlar üretilir

#### Avantajları
- Paralel işlem yapabilme (çok hızlı)
- Uzun cümleleri daha iyi anlama
- Kelimeler arası ilişkileri iyi yakalama

#### Dezavantajları
- Yüksek hesaplama gücü gerektirir
- Karmaşık bir yapıya sahiptir
- Eğitim için çok veri gerektirir

#### Avantajları
- Paralel işlem yeteneği
- Uzun mesafeli bağımlılıkları yakalama
- Ölçeklenebilirlik
- Daha hızlı eğitim süresi

#### Uygulama Alanları
- Makine çevirisi
- Metin özetleme
- Dil modelleme
- Soru cevaplama sistemleri

## 4. BERT ve GPT Modelleri

### 4.1 BERT (Bidirectional Encoder Representations from Transformers)
#### Temel Özellikleri
- Çift yönlü bağlam anlama
- Masked Language Model (MLM)
- Next Sentence Prediction (NSP)
- WordPiece tokenization

#### BERT Mimarisi
- Transformer encoder katmanları
- Multi-head attention
- Positional embeddings
- Segment embeddings

```python
from transformers import BertTokenizer, BertModel

# BERT model ve tokenizer yükleme
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Örnek kullanım
text = "NLP harika bir alan!"
encoded = tokenizer(text, return_tensors='pt')
outputs = model(**encoded)
```

### 4.2 GPT (Generative Pre-trained Transformer)
#### Temel Özellikleri
- Tek yönlü (left-to-right) dil modeli
- Autoregressive yapı
- BPE (Byte Pair Encoding) tokenization
- Büyük ölçekli dil modelleme

#### GPT Mimarisi
- Transformer decoder katmanları
- Masked self-attention
- Büyük parametre sayısı
- Scaling özellikleri

```python
from transformers import GPT2Tokenizer, GPT2Model

# GPT model ve tokenizer yükleme
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Metin üretimi örneği
input_text = "Yapay zeka"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
```

### 4.3 Transfer Learning ve Fine-tuning
#### Pre-training Aşaması
- Büyük veri setleri üzerinde eğitim
- Genel dil anlama
- Self-supervised learning
- Domain-agnostic öğrenme

#### Fine-tuning Teknikleri
- Task-specific adaptasyon
- Learning rate optimizasyonu
- Gradient accumulation
- Layer freezing stratejileri

```python
from transformers import BertForSequenceClassification

# Fine-tuning için model hazırlama
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Fine-tuning parametreleri
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5
)
```

### 4.4 Tokenization Stratejileri
#### WordPiece Tokenization
- Alt kelime birimleri
- Bilinmeyen kelime yönetimi
- Vocabulary optimizasyonu

#### BPE (Byte Pair Encoding)
- Karakter seviyesi tokenization
- Sık kullanılan alt birimler
- Adaptif vocabulary

```python
# Tokenization örnekleri
text = "Derin öğrenme ile doğal dil işleme"

# BERT tokenization
bert_tokens = tokenizer.tokenize(text)

# Alt kelime analizi
token_ids = tokenizer.encode(text)
decoded = tokenizer.decode(token_ids)
```

## 5. Uygulama Alanları
- Metin Sınıflandırma
- Duygu Analizi
- Makine Çevirisi
- Soru Cevaplama

## 6. En İyi Pratikler
- Veri Ön İşleme
- Model Seçimi
- Hiperparametre Optimizasyonu
- Model Değerlendirme

## 7. Sonuç
NLP alanındaki derin öğrenme modelleri, basit Word Embeddings'lerden karmaşık Transformer mimarilerine kadar geniş bir yelpazede gelişim göstermiştir. RNN'ler ve türevleri (LSTM, GRU) sıralı veri işlemede önemli başarılar elde etmiş, ancak Transformer mimarisi ile birlikte alan yeni bir boyut kazanmıştır. BERT ve GPT gibi modern modeller, transfer learning yaklaşımıyla birlikte, doğal dil işleme görevlerinde state-of-the-art sonuçlar elde etmeyi mümkün kılmıştır. Bu gelişmeler, metin sınıflandırma, makine çevirisi, duygu analizi ve soru cevaplama gibi pratik uygulamalarda önemli ilerlemeler sağlamıştır.