# Tekrarlayan Sinir Ağları (Recurrent Neural Networks - RNN)

## 📍 Bölüm Haritası
- Önceki Bölüm: [CNN Mimarisi](01-CNN.md)
- Sonraki Bölüm: [LSTM ve GRU](03-LSTM-GRU.md)
- Tahmini Süre: 5-6 saat
- Zorluk Seviyesi: 🟡 Orta

## 🎯 Hedefler
- RNN mimarisini ve çalışma prensibini anlama
- Sıralı veri işleme yöntemlerini öğrenme
- Vanishing gradient problemini kavrama
- Temel RNN uygulamaları geliştirme

## 🎯 Öz Değerlendirme
- [ ] RNN mimarisini açıklayabiliyorum
- [ ] Sıralı veri işleyebiliyorum
- [ ] Vanishing gradient problemini anlayabiliyorum
- [ ] RNN tabanlı modeller geliştirebiliyorum

## 🚀 Mini Projeler
1. Metin Üretimi
   - Karakter seviyesinde RNN
   - Shakespeare metinleri
   - Sıcaklık parametresi ayarlama

2. Zaman Serisi Tahmini
   - Hava durumu verisi
   - Çok değişkenli tahmin
   - Performans optimizasyonu

## 📑 Ön Koşullar
- Temel sinir ağları bilgisi
- Python ve derin öğrenme framework'leri
- Sıralı veri yapıları
- Optimizasyon teknikleri

## 🔑 Temel Kavramlar
1. Sıralı Veri İşleme
2. Gizli Durum (Hidden State)
3. Geri Yayılım (Backpropagation Through Time)
4. Vanishing Gradient

## Giriş
> Zorluk Seviyesi: 🟡 Orta

Tekrarlayan Sinir Ağları, sıralı verileri (metin, zaman serileri, vb.) işlemek için tasarlanmış özel bir yapay sinir ağı türüdür.

### Neden RNN?
- Sıralı verilerdeki bağımlılıkları yakalayabilme
- Değişken uzunlukta girdi işleyebilme
- Geçmiş bilgiyi kullanabilme

![RNN Basic Structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

## RNN'in Temel Yapısı
> �� İpucu: RNN'ler aynı ağırlıkları her zaman adımında tekrar kullanır

### 1. Basit RNN Hücresi
```python
import tensorflow as tf

class SimpleRNN(tf.keras.Model):
    def __init__(self, hidden_size):
        super(SimpleRNN, self).__init__()
        self.rnn = tf.keras.layers.SimpleRNN(
            hidden_size,
            return_sequences=True,
            return_state=True
        )
        
    def call(self, x, initial_state=None):
        # x shape: (batch_size, timesteps, input_dim)
        return self.rnn(x, initial_state=initial_state)
```

### 2. Gizli Durum (Hidden State)
```python
def init_hidden(self, batch_size):
    return tf.zeros([batch_size, self.hidden_size])
```

## RNN'in Çalışma Prensibi

### 1. İleri Yayılım
![RNN Forward Pass](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png)

```python
# Tek zaman adımı için
@tf.function
def rnn_cell_forward(x_t, h_prev, Wx, Wh, b):
    # Giriş ve önceki gizli durum birleştiriliyor
    concat = tf.concat([h_prev, x_t], axis=1)
    
    # Ağırlıklarla çarpım ve bias ekleme
    z = tf.matmul(concat, tf.concat([Wh, Wx], axis=0)) + b
    
    # Aktivasyon fonksiyonu
    h_next = tf.tanh(z)
    
    return h_next
```

### 2. Zaman İçinde Geri Yayılım (BPTT)
```python
@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    # Gradient clipping
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
```

## RNN Türleri

### 1. Çift Yönlü RNN (Bidirectional RNN)
```python
bidirectional_rnn = tf.keras.layers.Bidirectional(
    tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True)
)
```

![Bidirectional RNN](https://miro.medium.com/max/1400/1*6QnPUSv_t9BY9Fv8_aLb-Q.png)

### 2. Çok Katmanlı RNN
```python
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True),
    tf.keras.layers.SimpleRNN(hidden_size)
])
```

## RNN Uygulamaları

### 1. Metin Sınıflandırma
```python
class TextRNN(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(TextRNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.SimpleRNN(hidden_size)
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length)
        x = self.embedding(inputs)
        x = self.rnn(x)
        return self.dense(x)
```

### 2. Dizi Üretimi (Sequence Generation)
```python
def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        
        # Kategorik dağılımdan örnek alma
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0]
        
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    
    return start_string + ''.join(text_generated)
```

### 3. Metin Özetleme
```python
def build_seq2seq_summarizer():
    # Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(None,))
    enc_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(enc_emb)
    
    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    
    # Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    dec_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # Attention
    attention = tf.keras.layers.Attention()
    context_vector = attention([decoder_outputs, encoder_outputs])
    
    decoder_combined_context = tf.keras.layers.Concatenate(axis=-1)([context_vector, decoder_outputs])
    output = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder_combined_context)
    
    return tf.keras.Model([encoder_inputs, decoder_inputs], output)
```

### 4. Müzik Üretimi
```python
def build_music_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(256, input_shape=(sequence_length, num_notes), return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(512, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_notes, activation='softmax')
    ])
    return model
```

### 5. Zaman Serisi Anomali Tespiti
```python
def build_anomaly_detector():
    inputs = tf.keras.layers.Input(shape=(lookback, n_features))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    reconstruction = tf.keras.layers.Dense(n_features)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=reconstruction)
```

### 6. Duygu Analizi
```python
def build_sentiment_analyzer():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
```

## RNN'in Sorunları

### 1. Gradyan Kaybı/Patlaması
```python
# Gradient clipping örneği
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
```

### 2. Uzun Vadeli Bağımlılıklar
```python
# LSTM kullanımı
lstm_layer = tf.keras.layers.LSTM(hidden_size)
```

## Örnek: Karakter Seviyesi Dil Modeli

```python
class CharRNNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(
            rnn_units,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, states=None, return_state=False):
        x = self.embedding(inputs)
        if states is None:
            states = self.rnn.get_initial_state(x)
        x, states = self.rnn(x, initial_state=states)
        x = self.dense(x)
        
        if return_state:
            return x, states
        return x
```

## Model Eğitimi

```python
@tf.function
def train_step(inputs, targets, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                targets, predictions, from_logits=True))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss

EPOCHS = 10
for epoch in range(EPOCHS):
    hidden = None
    total_loss = 0
    
    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target, model, optimizer)
        total_loss += loss
        
        if batch_n % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch_n} Loss {loss:.4f}')
    
    print(f'Epoch {epoch+1} Loss {total_loss/batch_n:.4f}')
```

## Alıştırmalar

1. Basit bir RNN modeli oluşturun:
   - Karakter seviyesi dil modeli
   - Shakespeare metinleri üzerinde eğitim
   - Metin üretimi deneyin

2. Farklı RNN türlerini karşılaştırın:
   - Tek yönlü vs Çift yönlü
   - Tek katman vs Çok katman
   - Performans analizi yapın

3. Gradyan problemlerini gözlemleyin:
   - Farklı sekans uzunlukları deneyin
   - Gradyan değerlerini görselleştirin
   - Gradient clipping etkisini inceleyin

## 📚 Önerilen Kaynaklar
- [Colah's Blog - Understanding RNNs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Stanford CS224n NLP Dersleri](http://web.stanford.edu/class/cs224n/)
- [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. Karakter seviyesi metin üretimi
2. Basit zaman serisi tahmini

### Orta Seviye
1. Duygu analizi uygulaması
2. Çift yönlü RNN implementasyonu

### İleri Seviye
1. Attention mekanizması ekleme
2. Custom RNN hücresi tasarlama

## Görselleştirme Örnekleri

### 1. Attention Haritaları
```python
def plot_attention_weights(text, attention_weights):
    # Attention ağırlıklarını görselleştir
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_weights, cmap='viridis')
    
    # Eksenlere kelimeleri yerleştir
    plt.xticks(range(len(text)), text, rotation=45)
    plt.yticks(range(len(text)), text)
    
    plt.colorbar()
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.show()
```

### 2. Hidden State Dinamikleri
```python
def visualize_hidden_dynamics(model, input_sequence):
    # Hidden state'leri topla
    states = []
    h = None
    
    for t in range(len(input_sequence)):
        x = input_sequence[t:t+1]
        output, h = model(x, initial_state=h)
        states.append(h.numpy())
    
    # PCA ile boyut azaltma
    states = np.array(states).reshape(len(states), -1)
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states)
    
    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.scatter(states_2d[:, 0], states_2d[:, 1], c=range(len(states_2d)))
    plt.colorbar(label='Time Step')
    plt.title('Hidden State Dynamics (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
```

### 3. Karakter Tahmin Dağılımları
```python
def plot_prediction_distributions(model, text_seed, num_chars=50):
    # Her adımda tahmin dağılımlarını topla
    distributions = []
    current_text = text_seed
    
    for _ in range(num_chars):
        # Tahmin
        x = tf.convert_to_tensor([char2idx[c] for c in current_text])
        x = tf.expand_dims(x, 0)
        predictions = model(x)
        next_char_dist = predictions[0, -1].numpy()
        distributions.append(next_char_dist)
        
        # Bir sonraki karakter
        next_char_idx = np.random.choice(len(char2idx), p=next_char_dist)
        current_text += idx2char[next_char_idx]
    
    # Görselleştirme
    plt.figure(figsize=(15, 8))
    plt.imshow(np.array(distributions).T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.xlabel('Generation Step')
    plt.ylabel('Character')
    plt.title('Character Prediction Distributions Over Time')
    plt.yticks(range(len(char2idx)), list(char2idx.keys()))
    plt.show()
```

### 4. Gradyan Akışı
```python
def visualize_gradient_flow(model):
    # Gradyanları hesapla
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, outputs)
    
    grads = tape.gradient(loss, model.trainable_variables)
    
    # Gradyanları katmanlara göre görselleştir
    plt.figure(figsize=(10, 6))
    for i, (grad, var) in enumerate(zip(grads, model.trainable_variables)):
        grad_mean = tf.reduce_mean(tf.abs(grad)).numpy()
        plt.bar(i, grad_mean)
    
    plt.xticks(range(len(grads)), [var.name for var in model.trainable_variables], 
               rotation=45)
    plt.ylabel('Mean Gradient Magnitude')
    plt.title('Gradient Flow Across Layers')
    plt.tight_layout()
    plt.show()
```

### 5. Embedding Görselleştirme
```python
def visualize_word_embeddings(model, vocab, num_words=100):
    # Embedding matrisini al
    embedding_layer = model.get_layer('embedding')
    weights = embedding_layer.get_weights()[0]
    
    # t-SNE ile boyut azaltma
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(weights[:num_words])
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    # Kelimeleri ekle
    for i, word in enumerate(list(vocab.keys())[:num_words]):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title('Word Embeddings t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()
``` 