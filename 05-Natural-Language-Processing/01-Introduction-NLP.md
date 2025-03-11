# Doğal Dil İşleme (NLP)

Doğal Dil İşleme (Natural Language Processing - NLP), bilgisayarların insan dilini anlama, işleme ve üretme yeteneğini geliştirmeyi amaçlayan yapay zeka alanıdır.

## NLP'nin Temel Bileşenleri

### 1. Metin Ön İşleme (Text Preprocessing)

#### Tokenization
Metni anlamlı parçalara (token) ayırma işlemidir. Tokenization, NLP'nin en temel adımlarından biridir ve metnin daha detaylı analiz edilebilmesi için gereklidir.

Tokenization çeşitleri:
- Kelime Tokenization: Metni kelimelere ayırır
- Cümle Tokenization: Metni cümlelere ayırır
- Karakter Tokenization: Metni karakterlere ayırır
- Alt Kelime (Subword) Tokenization: Kelimeleri daha küçük anlamlı parçalara böler

Tokenization'ın önemi:
- Metin analizinin ilk adımıdır
- Daha ileri NLP işlemleri için temel oluşturur
- Dil modellerinin metni anlamasını sağlar
- Veri temizleme sürecinin önemli bir parçasıdır

> [!IMPORTANT]
> NLP işlemleri için `nltk` kütüphanesini kurmak gerekebilir. Aşağıdaki kod ile kurabilirsiniz:

```py
pip install nltk
```

```python
from nltk.tokenize import word_tokenize, sent_tokenize

# Kelime tokenization
text = "NLP çok önemli bir alandır!"
word_tokens = word_tokenize(text)
print(word_tokens)  # ['NLP', 'çok', 'önemli', 'bir', 'alandır', '!']

# Cümle tokenization
text = "NLP önemlidir. Yapay zeka gelişiyor."
sent_tokens = sent_tokenize(text)
print(sent_tokens)  # ['NLP önemlidir.', 'Yapay zeka gelişiyor.']
```
> [!NOTE]
> NLP işlemlerinde bazı işlemler için alt kütüphaneler gerekebilir. Kod hata verirse açıklamalarda hangi paketin kurulması gerektiği verilmektedir.

#### Stopwords (Durma Kelimeleri)

Stopwords (durma kelimeleri), bir metindeki anlamı doğrudan etkilemeyen, sık kullanılan yardımcı kelimelerdir. Bu kelimeler genellikle:
- Bağlaçlar (ve, veya, ile)
- Zamirler (ben, sen, o)
- Edatlar (için, gibi, kadar)
- Artikeller (bir, bu, şu)
gibi kelimelerden oluşur.

Stopwords'lerin çıkarılmasının avantajları:
- Metin işleme hızını artırır
- Depolama alanından tasarruf sağlar
- Analiz sonuçlarının daha anlamlı olmasını sağlar
- Model performansını iyileştirir

NLTK ile Türkçe stopwords kullanımı:
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Türkçe stopwords listesini yükleme
stop_words = set(stopwords.words('turkish'))

# Örnek metin
text = "Bu cümle bir örnek olarak yazılmıştır ve stopwords temizlenecektir"
tokens = word_tokenize(text)

# Stopwords temizleme
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Orijinal:", tokens)
print("Filtrelenmiş:", filtered_tokens)
```
> [!IMPORTANT]
> Stopwords temizliği için `nltk` kütüphanesinin `punk_tab` parçasını indirmeniz gerekir. Aşağıdaki kod ile indirebilirsiniz

```python
import nltk
nltk.download('punkt_tab')
```

Özel stopwords listesi oluşturma:
```python
# Kendi stopwords listenizi oluşturma
custom_stopwords = stop_words.union({'ek', 'kelime'})

# Özel listeyle filtreleme
filtered_custom = [word for word in tokens if word.lower() not in custom_stopwords]
```


#### Lemmatization ve Stemming

Lemmatization ve Stemming, kelimeleri kök veya temel formlarına indirgeme işlemleridir. Bu işlemler, metin analizi ve NLP çalışmalarında çok önemli bir rol oynar.

##### Stemming
Stemming, kelimelerin sonundaki ekleri kurallı bir şekilde kaldırarak kök formuna ulaşmaya çalışır.

Özellikleri:
- Hızlıdır
- Basit kurallara dayanır
- Sonuçlar her zaman gerçek kelime olmayabilir

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Örnek kullanım
words = ["running", "runs", "runner"]
stemmed = [stemmer.stem(word) for word in words]
print(stemmed)  # ['run', 'run', 'run']
```

##### Lemmatization
Lemmatization, kelimelerin sözlük anlamlarını ve dilbilgisi kurallarını dikkate alarak kök formuna ulaşır.

Özellikleri:
- Daha yavaştır ama daha doğrudur
- Morfolojik analiz yapar
- Sonuçlar her zaman anlamlı kelimelerdir

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Örnek kullanım
words = ["better", "running", "cars"]
lemmatized = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized)  # ['good', 'run', 'car']

# POS tag ile kullanım
print(lemmatizer.lemmatize('better', pos='a'))  # 'good'
print(lemmatizer.lemmatize('running', pos='v'))  # 'run'
print(lemmatizer.lemmatize('cars', pos='n'))  # 'car'
```
> [!IMPORTANT]
> WordNetLemmatizer kullanmak için `nltk` kütüphanesinin `wordnet` parçasını indirmeniz gerekir. Aşağıdaki kod ile indirebilirsiniz

```python
import nltk
nltk.download('wordnet')
```

Türkçe için Lemmatization örneği:
```python
import zeyrek
analyzer = zeyrek.MorphAnalyzer()

# Örnek kullanım
word = "kitaplardan"
analysis = analyzer.lemmatize(word)
print(analysis)  # [('kitap', ...)]
```
> [!IMPORTANT]
> Zeyrek kütüphanesini kurmak gerekebilir. Aşağıdaki kod ile kurabilirsiniz:

```py
pip install zeyrek
```

Kullanım alanları:
- Metin normalleştirme
- Arama motorları
- Metin sınıflandırma
- Duygu analizi
- Doğal dil işleme modelleri

### 2. Sözdizimsel Analiz (Syntactic Analysis)

#### POS Tagging (Sözcük Türü Etiketleme)

POS (Part of Speech) Tagging, bir metindeki her kelimenin dilbilgisel görevini (isim, fiil, sıfat, zarf vb.) otomatik olarak belirleme işlemidir.

Temel POS kategorileri:
- Noun (İsim): Varlık, nesne, kavram adları
- Verb (Fiil): Eylem bildiren kelimeler
- Adjective (Sıfat): İsimleri niteleyen kelimeler
- Adverb (Zarf): Fiilleri niteleyen kelimeler
- Pronoun (Zamir): İsimlerin yerini tutan kelimeler
- Preposition (Edat): İlişki kuran kelimeler

POS Tagging'in önemi:
- Sözdizimsel analiz için temeldir
- Anlam belirsizliğini gidermeye yardımcı olur
- Makine çevirisi kalitesini artırır
- Metin madenciliği uygulamalarını geliştirir

NLTK ile POS Tagging örneği:
```python
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Örnek metin
text = "Python ile doğal dil işleme öğreniyorum"
tokens = word_tokenize(text)

# POS tagging uygulama
pos_tags = pos_tag(tokens)
print(pos_tags)  # [('Python', 'NNP'), ('ile', 'IN'), ...]
```

#### Parsing (Ayrıştırma)

Parsing, doğal dil işlemede cümlelerin dilbilgisel yapısını analiz etme ve çözümleme işlemidir. Bu işlem, cümlenin sözdizimsel yapısını bir ağaç yapısına dönüştürür.

Temel Parsing türleri:
- Constituency Parsing: Cümleyi hiyerarşik yapı birimlerine ayırır
- Dependency Parsing: Kelimeler arasındaki ilişkileri belirler
- Shallow Parsing: Yüzeysel sözdizimsel analiz yapar

NLTK ile Constituency Parsing örneği:
```python
from nltk import CFG
from nltk.parse import RecursiveDescentParser

# Basit bir dilbilgisi tanımlama
grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N | N
    VP -> V NP
    Det -> 'the' | 'a'
    N -> 'cat' | 'dog'
    V -> 'chased' | 'saw'
""")

parser = RecursiveDescentParser(grammar)
sentence = "the cat chased a dog".split()
for tree in parser.parse(sentence):
    print(tree)
```

Spacy ile Dependency Parsing:
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog")

# Bağımlılık ilişkilerini gösterme
for token in doc:
    print(f"{token.text:>12} --{token.dep_}--> {token.head.text}")
```
> [!IMPORTANT]
> Spacy kütüphanesini ve alt kütüphanelerini kurmak gerekebilir. Aşağıdaki kodları sırayla çalıştırarak kurabilirsiniz:

```py
pip install zeyrek
!python -m spacy download en_core_web_sm
```

Parsing'in kullanım alanları:
- Gramer kontrolü
- Makine çevirisi
- Bilgi çıkarımı
- Cümle yapısı analizi
- Doğal dil anlama sistemleri

### 3. Anlambilimsel Analiz (Semantic Analysis)

#### Named Entity Recognition (NER)

Named Entity Recognition (NER), metinlerdeki özel isimleri, tarihleri, lokasyonları ve diğer önemli varlıkları otomatik olarak tespit etme ve sınıflandırma işlemidir.

Temel NER kategorileri:
- PERSON: Kişi isimleri
- ORG: Organizasyon isimleri
- LOC: Lokasyon isimleri
- DATE: Tarih ifadeleri
- TIME: Zaman ifadeleri
- MONEY: Para birimleri
- GPE: Ülke, şehir gibi coğrafi-politik varlıklar

NER'in önemi:
- Bilgi çıkarımı için temeldir
- Metin analizi ve sınıflandırmada kullanılır
- Arama motorları optimizasyonunda önemlidir
- Doküman indeksleme ve filtrelemede kullanılır

Spacy ile NER örneği:
```python
import spacy

# İngilizce model yükleme
nlp = spacy.load("en_core_web_sm")

# Örnek metin
text = "John Doe gave a presentation at Microsoft's New York office on March 15, 2024."
doc = nlp(text)

# Varlıkları belirleme
for ent in doc.ents:
    print(f"Entity: {ent.text}\nType: {ent.label_}\n")
```

NLTK ile NER örneği:
```python
from nltk import ne_chunk
from nltk import pos_tag
from nltk import word_tokenize

text = "Steve Jobs Apple'ı 1976'da kurdu."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)
print(entities)
```
> [!IMPORTANT]
> WordNetLemmatizer kullanmak için `nltk` kütüphanesinin `maxent_ne_chunker_tab` ve `` parçalarını indirmeniz gerekir. Aşağıdaki kodlarla sıra ile indirebilirsiniz:

```python
import nltk
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
```

NER kullanım alanları:
- Müşteri hizmetleri sistemleri
- Otomatik doküman sınıflandırma
- İçerik analizi ve filtreleme
- Sosyal medya analizi
- Biyomedikal araştırmalar

#### Word Sense Disambiguation (Kelime Anlam Belirsizliği Giderme)

Word Sense Disambiguation (WSD), çok anlamlı kelimelerin cümle içindeki bağlamına göre doğru anlamını belirleme işlemidir. Bu işlem, NLP'nin en zorlu görevlerinden biridir.

Temel WSD yaklaşımları:
- Bilgi tabanlı yöntemler (WordNet gibi)
- Denetimli öğrenme yöntemleri
- Denetimsiz öğrenme yöntemleri
- Hibrit yaklaşımlar

WSD'nin önemi:
- Makine çevirisinde doğruluğu artırır
- Bilgi çıkarımı kalitesini yükseltir
- Metin anlama sistemlerini geliştirir
- Arama motoru sonuçlarını iyileştirir

NLTK ile basit WSD örneği:
```python
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

sentences = [
    "Her face showed a mixture of surprise and joy.", 
    "Her face displayed a wide range of emotions.",  
    "She had to face her fears during the presentation.",   
    "The face of the building was covered in beautiful murals.",  
]

# 'face' kelimesinin anlamlarını bulma
for sentence in sentences:
    sense = lesk(word_tokenize(sentence), "face")
    print(f"Meaning: {sense}")

# WordNet'te 'face' kelimesinin anlamlarını kontrol etme
synsets = wn.synsets("face")
print(f"WordNet Synsets for 'face': {synsets}")
```

Modern WSD yaklaşımları:
- BERT tabanlı modeller
- Transformers mimarisi
- Bağlamsal embedding'ler
- Transfer öğrenme

Türkçe WSD için özel durumlar:
- Çok anlamlılık
- Eş seslilik
- Mecaz anlamlar
- Bağlam bağımlılığı

```python
# Transformers ile WSD örneği
from transformers import pipeline
nlp = pipeline('fill-mask')

text = "She was so excited to see the <mask> at the concert."
results = nlp(text)
results
```
> [!IMPORTANT]
> transformers kütüphanesini ve alt kütüphanelerini kurmak gerekebilir. Aşağıdaki kodları sırayla çalıştırarak kurabilirsiniz:

```py
pip install transformers
pip install tf_keras
```

## NLP Uygulamaları

- Makine Çevirisi
- Metin Özetleme
- Duygu Analizi
- Chatbotlar
- Ses Tanıma
- Metin Üretme

## Günümüz NLP Teknolojileri

- BERT
- GPT
- Transformer modelleri
- Word2Vec
- LSTM ağları

## Zorluklar ve Çözümler

- Dil belirsizlikleri
- Bağlam anlama
- Çok dilli sistemler
- Veri kalitesi
## Sonuç

Doğal Dil İşleme (NLP), modern teknoloji dünyasının en önemli araştırma alanlarından biridir. Bu kapsamda öğrendiğimiz temel bileşenler:

- Metin Ön İşleme
    * Tokenization
    * Stopwords temizleme
    * Lemmatization ve Stemming

- Sözdizimsel Analiz
    * POS Tagging
    * Parsing işlemleri

- Anlambilimsel Analiz
    * Named Entity Recognition
    * Word Sense Disambiguation

Bu teknolojiler, chatbotlardan makine çevirisine, duygu analizinden metin özetlemeye kadar birçok uygulamada kullanılmaktadır. Günümüzde BERT, GPT gibi gelişmiş dil modelleri sayesinde NLP teknolojileri giderek daha güçlenmekte ve insan-makine etkileşiminde yeni ufuklar açmaktadır. Dil işleme yeteneklerinin gelişmesi, yapay zeka sistemlerinin insanlarla daha doğal ve etkili iletişim kurmasını sağlamaktadır.