# Doğal Dil İşleme (Natural Language Processing)

## 🎯 Hedefler
- Temel NLP görevlerini anlama ve uygulama
- Modern NLP modellerini kullanabilme
- Metin verisi ön işleme tekniklerini öğrenme
- Transfer learning ile NLP modelleri geliştirme

## 📑 Ön Koşullar
- RNN, LSTM ve Transformer mimarileri bilgisi
- Python ve TensorFlow/PyTorch deneyimi
- Temel metin işleme kavramları
- Tokenizasyon ve embedding kavramları

## 🔑 Temel Kavramlar
1. Tokenizasyon ve Vektörleştirme
2. Word Embeddings
3. Sequence Modeling
4. Attention Mekanizmaları
5. Fine-tuning ve Transfer Learning

## Veri Hazırlama
> Zorluk Seviyesi: 🟡 Orta

> �� İpucu: Metin ön işleme, NLP modellerinin performansını büyük ölçüde etkiler

## Metin Ön İşleme

### 1. Tokenization ve Padding
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Örnek veri
texts = [
    "Bu bir örnek cümledir",
    "NLP çok ilginç bir alan",
    "Derin öğrenme ile NLP yapalım"
]

# Tokenizer oluşturma
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Metinleri sayısal dizilere çevirme
sequences = tokenizer.texts_to_sequences(texts)

# Padding işlemi
padded = pad_sequences(sequences, maxlen=10, padding='post')
```

### 2. Word Embeddings
```python
# Embedding katmanı
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Temel NLP Görevleri

### 1. Duygu Analizi
```python
# IMDB veri seti
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=10000
)

# Model oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Model derleme
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 2. Metin Sınıflandırma
```python
# BERT ile metin sınıflandırma
import tensorflow_hub as hub
import tensorflow_text as text

# BERT modeli yükleme
bert_preprocess = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Model oluşturma
def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation='softmax')(net)
    
    return tf.keras.Model(text_input, net)
```

### 3. Metin Üretimi
```python
# Karakter seviyesi RNN
class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                      return_sequences=True,
                                      return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False):
        x = self.embedding(inputs)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states)
        x = self.dense(x)

        if return_state:
            return x, states
        return x
```

## Modern NLP Teknikleri

### 1. Transformer Kullanımı
```python
# Transformer encoder
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
```

### 2. Transfer Learning
```python
# HuggingFace modeli kullanımı
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

# Model ve tokenizer yükleme
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Fine-tuning için model hazırlama
input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

outputs = model([input_ids, attention_mask])
```

## Örnek Uygulamalar

### 1. Spam Tespiti
```python
def build_spam_detector():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 32),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
```

### 2. Soru Cevaplama
```python
def build_qa_model():
    question_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    context_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    
    preprocessed_question = bert_preprocess(question_input)
    preprocessed_context = bert_preprocess(context_input)
    
    outputs = bert_encoder([preprocessed_question, preprocessed_context])
    
    start_logits = tf.keras.layers.Dense(1)(outputs.last_hidden_state)
    end_logits = tf.keras.layers.Dense(1)(outputs.last_hidden_state)
    
    return tf.keras.Model(
        inputs=[question_input, context_input],
        outputs=[start_logits, end_logits]
    )
```

### 3. Çoklu Dil Desteği
```python
def build_multilingual_classifier():
    # XLM-RoBERTa modelini yükle
    xlm_roberta = TFAutoModel.from_pretrained('xlm-roberta-base')
    
    # Model mimarisi
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    
    # XLM-RoBERTa çıktıları
    outputs = xlm_roberta(input_ids, attention_mask=attention_mask)[0]
    
    # Sınıflandırma katmanları
    x = tf.keras.layers.GlobalAveragePooling1D()(outputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
```

### 4. Metin Özetleme
```python
def build_text_summarizer():
    # T5 modelini yükle
    t5_model = TFT5ForConditionalGeneration.from_pretrained('t5-base')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    class Summarizer:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
        
        def summarize(self, text, max_length=150):
            # Girdi metni hazırlama
            inputs = self.tokenizer.encode(
                "summarize: " + text,
                return_tensors='tf',
                max_length=512,
                truncation=True
            )
            
            # Özet oluşturma
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=40,
                num_beams=4,
                no_repeat_ngram_size=2
            )
            
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return Summarizer(t5_model, t5_tokenizer)
```

### 5. Duygu Analizi
```python
def build_sentiment_analyzer():
    # BERT tabanlı model
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    
    # Giriş katmanları
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    
    # BERT çıktıları
    sequence_output = bert(input_ids, attention_mask=attention_mask)[0]
    
    # Özel katmanlar
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Çoklu duygu çıktısı (örn: pozitif, negatif, nötr)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    
    # Özel metrikler
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model
```

### 6. Soru-Cevap Sistemi
```python
def build_qa_system():
    # BERT tabanlı soru-cevap modeli
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_qa = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')
    
    class QuestionAnswering:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
        
        def answer_question(self, context, question):
            # Girdiyi hazırla
            inputs = self.tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                return_tensors='tf',
                max_length=512,
                truncation=True
            )
            
            # Tahmin
            outputs = self.model(inputs)
            answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
            answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0]
            
            # Cevabı decode et
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            answer = self.tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end+1])
            
            return answer
    
    return QuestionAnswering(bert_qa, tokenizer)
```

## 📚 Önerilen Kaynaklar
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [TensorFlow Text Tutorial](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
- [spaCy Course](https://course.spacy.io/)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. Duygu analizi uygulaması
2. Basit metin sınıflandırma

### Orta Seviye
1. Çoklu dil desteği ekleme
2. Named Entity Recognition sistemi

### İleri Seviye
1. Soru-cevap sistemi geliştirme
2. Metin özetleme modeli oluşturma

## Alıştırmalar
1. Duygu Analizi:
   - Twitter verisi üzerinde duygu analizi yapın
   - BERT ve LSTM modellerini karşılaştırın
   - Model performansını artırın

2. Metin Üretimi:
   - Şiir/hikaye veri seti toplayın
   - Karakter seviyesi RNN oluşturun
   - Farklı sıcaklık değerleriyle üretim yapın

3. Metin Sınıflandırma:
   - Kendi veri setinizi oluşturun
   - Transfer learning uygulayın
   - Confusion matrix analizi yapın

## Kaynaklar
1. [TensorFlow Text Tutorial](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
2. [BERT Guide](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
3. [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 