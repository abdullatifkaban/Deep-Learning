# LSTM ve GRU (Long Short-Term Memory & Gated Recurrent Unit)

## 📍 Bölüm Haritası
- Önceki Bölüm: [RNN](02-RNN.md)
- Sonraki Bölüm: [Transformers](04-Transformers.md)
- Tahmini Süre: 5-6 saat
- Zorluk Seviyesi: 🟡 Orta

## 🎯 Hedefler
- LSTM ve GRU mimarilerini anlama
- Uzun vadeli bağımlılıkları öğrenme
- Kapı mekanizmalarını kavrama
- Modern uygulamaları geliştirme

## �� Öz Değerlendirme
- [ ] LSTM ve GRU yapılarını açıklayabiliyorum
- [ ] Kapı mekanizmalarını anlayabiliyorum
- [ ] Farklı mimarileri karşılaştırabiliyorum
- [ ] Uygun mimariye karar verebiliyorum

## 🚀 Mini Projeler
1. Dil Modeli
   - LSTM tabanlı dil modeli
   - Farklı kapı yapıları
   - Performans analizi

2. Müzik Üretimi
   - MIDI dosya işleme
   - GRU tabanlı model
   - Stil transferi

## 📑 Ön Koşullar
- RNN mimarisi bilgisi
- Python ve derin öğrenme framework'leri
- Gradyan akışı kavramları
- Optimizasyon teknikleri

## �� Temel Kavramlar
1. Kapı Mekanizmaları
2. Hücre Durumu
3. Uzun Vadeli Bağımlılıklar
4. Gradyan Kontrolü

## Giriş
🟡 Orta

LSTM ve GRU, RNN'lerin uzun vadeli bağımlılıkları öğrenmedeki problemlerini çözmek için geliştirilmiş özel mimari yapılardır.

### Neden LSTM/GRU?
- Gradyan kaybı/patlaması problemini çözme
- Uzun vadeli bağımlılıkları yakalayabilme
- Seçici bilgi akışı kontrolü

## LSTM'in Yapısı
💡 İpucu: LSTM'in üç kapısı vardır: forget, input ve output

### 1. LSTM Hücresi

![LSTM Structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

1. **Forget Gate**: Hangi bilgilerin unutulacağına karar verir
2. **Input Gate**: Hangi yeni bilgilerin saklanacağına karar verir
3. **Cell State**: Uzun vadeli bilgiyi taşır
4. **Output Gate**: Hangi bilgilerin çıktı olarak verileceğine karar verir

### LSTM Implementasyonu

```python
import tensorflow as tf

# Basit LSTM modeli
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, 
                        input_shape=(timesteps, features)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Çift yönlü LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        input_shape=(timesteps, features)
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### LSTM Katmanı Özellikleri

```python
lstm_layer = tf.keras.layers.LSTM(
    units=64,                    # Gizli birim sayısı
    activation='tanh',           # Ana aktivasyon fonksiyonu
    recurrent_activation='sigmoid', # İç kapı aktivasyonları
    return_sequences=True,       # Tüm zaman adımları için çıktı
    return_state=False,          # Durum vektörlerini döndürme
    stateful=False,              # Durum hafızası
    dropout=0.2,                 # Giriş dropout oranı
    recurrent_dropout=0.2        # Tekrarlayan bağlantı dropout oranı
)
```

## GRU (Gated Recurrent Unit)

GRU, LSTM'in daha basit bir versiyonudur ve daha az parametre içerir.

### GRU'nun Yapısı

![GRU Structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

1. **Update Gate**: LSTM'deki forget ve input gate'lerin birleşimi
2. **Reset Gate**: Önceki durumun ne kadarının unutulacağını belirler

### GRU Implementasyonu

```python
# Basit GRU modeli
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True,
                       input_shape=(timesteps, features)),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Çift yönlü GRU
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64, return_sequences=True),
        input_shape=(timesteps, features)
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## LSTM vs GRU Karşılaştırması

```python
# Aynı problem için her iki model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dense(1)
])

# Model derleme
lstm_model.compile(optimizer='adam', loss='mse')
gru_model.compile(optimizer='adam', loss='mse')

# Eğitim ve karşılaştırma
lstm_history = lstm_model.fit(X_train, y_train, validation_split=0.2, epochs=100)
gru_history = gru_model.fit(X_train, y_train, validation_split=0.2, epochs=100)

# Performans karşılaştırma grafiği
plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
plt.plot(gru_history.history['loss'], label='GRU Training Loss')
plt.legend()
plt.show()
```

## Örnek Uygulamalar

### 1. Metin Sınıflandırma
```python
# Metin sınıflandırma modeli
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

### 2. Zaman Serisi Tahmini
```python
# Zaman serisi modeli
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True,
                        input_shape=(n_steps, n_features)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])
```

### 3. Makine Çevirisi
```python
def build_nmt_model(src_vocab_size, tgt_vocab_size, embedding_dim=256):
    # Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(None,))
    encoder_embedding = tf.keras.layers.Embedding(src_vocab_size, embedding_dim)(encoder_inputs)
    
    # Bidirectional LSTM encoder
    encoder = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(embedding_dim, return_state=True)
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_embedding)
    
    # Birleştirme
    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(tgt_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(embedding_dim*2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Attention mekanizması
    attention = tf.keras.layers.Attention()
    context_vector = attention([decoder_outputs, encoder_outputs])
    
    # Çıktı katmanı
    decoder_dense = tf.keras.layers.Dense(tgt_vocab_size, activation='softmax')
    outputs = decoder_dense(decoder_outputs)
    
    return tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
```

### 4. Finansal Tahmin
```python
def build_stock_predictor():
    model = tf.keras.Sequential([
        # Feature extraction
        tf.keras.layers.LSTM(128, return_sequences=True, 
                           input_shape=(lookback, n_features)),
        tf.keras.layers.Dropout(0.2),
        
        # Market pattern analysis
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        
        # Long-term dependencies
        tf.keras.layers.GRU(128, return_sequences=False),
        
        # Price prediction
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(prediction_horizon)  # Çoklu zaman adımı tahmini
    ])
    return model
```

### 5. Video Analizi
```python
def build_video_classifier():
    # CNN özellik çıkarıcı
    cnn_base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    cnn_base.trainable = False
    
    # Video sınıflandırma modeli
    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(cnn_base),
        tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()),
        
        # Temporal özellik çıkarma
        tf.keras.layers.LSTM(512, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(256),
        
        # Sınıflandırma
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### 6. Ses Tanıma
```python
def build_speech_recognizer():
    # Spektrogram girişi
    inputs = tf.keras.layers.Input(shape=(None, n_mels))
    
    # Özellik çıkarma
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Temporal modelleme
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(x)
    
    # CTC loss için çıktı
    outputs = tf.keras.layers.Dense(num_characters + 1, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

## Görselleştirme Örnekleri

### 1. Gate Aktivasyonları
```python
def visualize_gate_activations(model, input_sequence):
    # Gate aktivasyonlarını almak için özel model
    gate_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer('lstm').cell.input_gate,
            model.get_layer('lstm').cell.forget_gate,
            model.get_layer('lstm').cell.output_gate
        ]
    )
    
    # Gate aktivasyonlarını hesapla
    i_gate, f_gate, o_gate = gate_model.predict(input_sequence)
    
    # Görselleştirme
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(i_gate.T, aspect='auto', cmap='viridis')
    plt.title('Input Gate Activations')
    plt.xlabel('Time Step')
    plt.ylabel('Hidden Unit')
    
    plt.subplot(1, 3, 2)
    plt.imshow(f_gate.T, aspect='auto', cmap='viridis')
    plt.title('Forget Gate Activations')
    plt.xlabel('Time Step')
    
    plt.subplot(1, 3, 3)
    plt.imshow(o_gate.T, aspect='auto', cmap='viridis')
    plt.title('Output Gate Activations')
    plt.xlabel('Time Step')
    
    plt.tight_layout()
    plt.show()
```

### 2. Memory Cell Analizi
```python
def analyze_memory_cells(model, input_sequence):
    # Memory cell durumlarını almak için özel model
    memory_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('lstm').cell.state_h
    )
    
    # Memory cell durumlarını hesapla
    cell_states = memory_model.predict(input_sequence)
    
    # Seçili hücreleri görselleştir
    plt.figure(figsize=(12, 6))
    for i in range(min(5, cell_states.shape[-1])):
        plt.plot(cell_states[:, i], label=f'Cell {i}')
    
    plt.title('Memory Cell States Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cell State Value')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 3. LSTM vs GRU Karşılaştırması
```python
def compare_lstm_gru_performance(X_train, y_train, epochs=50):
    # LSTM ve GRU modelleri
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    
    gru_model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    
    # Modelleri derle
    lstm_model.compile(optimizer='adam', loss='mse')
    gru_model.compile(optimizer='adam', loss='mse')
    
    # Eğitim
    lstm_history = lstm_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
    gru_history = gru_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
    
    # Performans karşılaştırması
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history.history['loss'], label='LSTM Training')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Validation')
    plt.plot(gru_history.history['loss'], label='GRU Training')
    plt.plot(gru_history.history['val_loss'], label='GRU Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

### 4. Uzun Vadeli Bağımlılık Analizi
```python
def analyze_long_term_dependencies(model, input_sequence, target_time_step):
    # Gradyan analizi için özel fonksiyon
    @tf.function
    def get_gradients(input_seq):
        with tf.GradientTape() as tape:
            tape.watch(input_seq)
            output = model(input_seq)
            target_output = output[:, target_time_step, :]
        return tape.gradient(target_output, input_seq)
    
    # Gradyanları hesapla
    grads = get_gradients(input_sequence)
    grad_magnitudes = tf.reduce_mean(tf.abs(grads), axis=-1)
    
    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.plot(grad_magnitudes[0])
    plt.title(f'Gradient Magnitudes for Target Time Step {target_time_step}')
    plt.xlabel('Input Time Step')
    plt.ylabel('Average Gradient Magnitude')
    plt.grid(True)
    plt.show()
```

### 5. Özellik Önem Analizi
```python
def visualize_feature_importance(model, X_test, feature_names):
    # SHAP değerleri hesapla
    import shap
    explainer = shap.DeepExplainer(model, X_test[:100])
    shap_values = explainer.shap_values(X_test[:1000])
    
    # Özellik önemlerini görselleştir
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values[0], X_test[:1000], feature_names=feature_names)
    plt.title('Feature Importance Analysis')
    plt.show()
```

## 📚 Önerilen Kaynaklar
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Illustrated Guide to LSTM's and GRU's](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn#lstm_layers)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. Basit zaman serisi tahmini
2. Duygu analizi uygulaması

### Orta Seviye
1. Müzik üretimi
2. Makine çevirisi

### İleri Seviye
1. Custom LSTM/GRU hücresi tasarlama
2. Attention mekanizması entegrasyonu 