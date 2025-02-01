# Transformers

## 🎯 Hedefler
- Transformer mimarisinin temel bileşenlerini anlama
- Self-attention mekanizmasının çalışma prensibini kavrama
- Encoder-Decoder yapısını öğrenme
- Modern Transformer modellerini (BERT, GPT) tanıma

## 📑 Ön Koşullar
- RNN ve LSTM mimarilerini anlama
- Python ve derin öğrenme framework'leri deneyimi
- Matris işlemleri ve lineer cebir temelleri
- Attention mekanizması hakkında temel bilgi

## �� Temel Kavramlar
1. Self-Attention
2. Multi-Head Attention
3. Positional Encoding
4. Layer Normalization
5. Residual Connections

## Giriş
> Zorluk Seviyesi: 🔴 İleri

Transformer mimarisi, "Attention is All You Need" makalesiyle tanıtılmış ve NLP alanında çığır açmış bir modeldir.

### Neden Transformers?
- Paralel işleme yeteneği
- Uzun mesafeli bağımlılıkları yakalayabilme
- Ölçeklenebilir mimari
- State-of-the-art performans

![Transformer Architecture](https://transformer-architecture.png)

## Transformer'ın Temel Bileşenleri
> 💡 İpucu: Self-attention, modelin herhangi bir konumdaki kelimenin diğer kelimelerle ilişkisini öğrenmesini sağlar

### 1. Attention Mekanizması
```python
import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask=None):
    # Attention(Q, K, V) = softmax(QK^T/√d_k)V
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights
```

### 2. Multi-Head Attention
```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
```

### 3. Positional Encoding
```python
def get_positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
```

## Transformer Mimarisi

### 1. Encoder
```python
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
```

### 2. Decoder
```python
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
```

## Örnek Uygulamalar

### 1. Makine Çevirisi
```python
# Transformer tabanlı çeviri modeli
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                             input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                             target_vocab_size, rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
```

### 2. BERT Kullanımı
```python
import tensorflow_hub as hub
import tensorflow_text as text

# BERT modeli yükleme
bert_preprocess = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Sınıflandırma modeli
def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    
    return tf.keras.Model(text_input, net)
```

### 3. Çoklu Dil Çevirisi
```python
def build_multilingual_transformer():
    # Dil kodlayıcı
    language_embedding = tf.keras.layers.Embedding(num_languages, d_model)
    
    # Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(max_length,))
    encoder_embedding = tf.keras.layers.Embedding(src_vocab_size, d_model)(encoder_inputs)
    encoder_positional = positional_encoding(max_length, d_model)
    
    x = encoder_embedding + encoder_positional
    
    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, dff, dropout_rate)
    
    # Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(max_length,))
    decoder_embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)(decoder_inputs)
    decoder_positional = positional_encoding(max_length, d_model)
    
    y = decoder_embedding + decoder_positional
    
    for _ in range(num_layers):
        y = transformer_decoder(y, x, d_model, num_heads, dff, dropout_rate)
    
    outputs = tf.keras.layers.Dense(tgt_vocab_size, activation='softmax')(y)
    
    return tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
```

### 4. Görüntü Transformers (ViT)
```python
def build_vision_transformer():
    # Patch embedding
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, channels))
    patches = tf.keras.layers.Conv2D(
        filters=d_model,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid'
    )(inputs)
    
    # Reshape patches
    batch_size = tf.shape(patches)[0]
    patches = tf.reshape(patches, [batch_size, -1, d_model])
    
    # Position embeddings
    positions = tf.range(start=0, limit=tf.shape(patches)[1], delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=d_model)(positions)
    x = patches + pos_embedding
    
    # Transformer blocks
    for _ in range(num_layers):
        x = transformer_encoder_block(x, d_model, num_heads, mlp_dim)
    
    # Classification head
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### 5. Belge Sınıflandırma
```python
def build_document_classifier():
    # BERT tabanlı model
    bert_encoder = TFBertModel.from_pretrained('bert-base-multilingual-cased')
    
    # Giriş katmanları
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    
    # BERT çıktıları
    sequence_output = bert_encoder(input_ids, attention_mask=attention_mask)[0]
    
    # Özel sınıflandırma katmanları
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
```

### 6. Soru-Cevap Sistemi
```python
def build_qa_transformer():
    # BERT encoder
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    
    # Giriş katmanları
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    
    # BERT çıktıları
    outputs = bert(input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids)
    
    # Başlangıç ve bitiş pozisyonları için çıktı katmanları
    start_logits = tf.keras.layers.Dense(1, name='start_logit')(outputs[0])
    start_logits = tf.keras.layers.Flatten()(start_logits)
    
    end_logits = tf.keras.layers.Dense(1, name='end_logit')(outputs[0])
    end_logits = tf.keras.layers.Flatten()(end_logits)
    
    start_probs = tf.keras.layers.Activation('softmax')(start_logits)
    end_probs = tf.keras.layers.Activation('softmax')(end_logits)
    
    return tf.keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=[start_probs, end_probs]
    )
```

## Görselleştirme Örnekleri

### 1. Multi-Head Attention Görselleştirme
```python
def visualize_attention_heads(model, input_text, layer_idx=0):
    # Attention ağırlıklarını al
    attention_weights = model.layers[layer_idx].attention.attention_weights
    
    # Tokenize et
    tokens = tokenizer.tokenize(input_text)
    
    # Her head için attention matrisini görselleştir
    num_heads = attention_weights.shape[1]
    fig, axes = plt.subplots(2, num_heads//2, figsize=(15, 8))
    axes = axes.flatten()
    
    for h in range(num_heads):
        im = axes[h].imshow(attention_weights[0, h], cmap='viridis')
        axes[h].set_xticks(range(len(tokens)))
        axes[h].set_yticks(range(len(tokens)))
        axes[h].set_xticklabels(tokens, rotation=45)
        axes[h].set_yticklabels(tokens)
        axes[h].set_title(f'Head {h+1}')
    
    plt.colorbar(im, ax=axes)
    plt.tight_layout()
    plt.show()
```

### 2. Positional Encoding Görselleştirme
```python
def visualize_positional_encoding(max_position, d_model):
    pos_encoding = positional_encoding(max_position, d_model)
    
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.colorbar(label='Value')
    plt.title('Positional Encoding Values')
    plt.show()
    
    # Sinüzoidal fonksiyonları görselleştir
    plt.figure(figsize=(12, 4))
    for i in range(4):
        plt.plot(pos_encoding[0, :, i], label=f'dim {i}')
    plt.legend()
    plt.title('Positional Encoding Sinusoids')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
```

### 3. Layer-wise Analiz
```python
def analyze_layer_outputs(model, input_text):
    # Her katmanın çıktısını al
    layer_outputs = []
    current_input = input_text
    
    for layer in model.layers:
        current_output = layer(current_input)
        layer_outputs.append(current_output)
        current_input = current_output
    
    # PCA ile boyut azaltma ve görselleştirme
    plt.figure(figsize=(15, 5))
    for i, output in enumerate(layer_outputs):
        # Output'u 2D'ye indir
        output_2d = PCA(n_components=2).fit_transform(output[0])
        
        plt.subplot(1, len(layer_outputs), i+1)
        plt.scatter(output_2d[:, 0], output_2d[:, 1])
        plt.title(f'Layer {i+1}')
    
    plt.tight_layout()
    plt.show()
```

### 4. Attention Flow Analizi
```python
def visualize_attention_flow(model, input_text, target_token_idx):
    # Tüm katmanlardan attention ağırlıklarını topla
    attention_weights = []
    for layer in model.layers:
        if hasattr(layer, 'attention'):
            weights = layer.attention.attention_weights
            attention_weights.append(weights)
    
    # Attention akışını görselleştir
    plt.figure(figsize=(12, len(attention_weights) * 3))
    
    for i, weights in enumerate(attention_weights):
        plt.subplot(len(attention_weights), 1, i+1)
        
        # Target token'a olan attention'ları göster
        plt.plot(weights[0, :, target_token_idx])
        plt.title(f'Layer {i+1} Attention Flow')
        plt.xlabel('Source Token Index')
        plt.ylabel('Attention Weight')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

### 5. Cross-Attention Görselleştirme
```python
def visualize_cross_attention(model, source_text, target_text):
    # Encoder-Decoder attention ağırlıklarını al
    cross_attention_weights = model.decoder.cross_attention.attention_weights
    
    # Tokenize
    source_tokens = tokenizer.tokenize(source_text)
    target_tokens = tokenizer.tokenize(target_text)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    plt.imshow(cross_attention_weights[0], cmap='viridis')
    
    plt.xticks(range(len(source_tokens)), source_tokens, rotation=45)
    plt.yticks(range(len(target_tokens)), target_tokens)
    
    plt.colorbar(label='Attention Weight')
    plt.title('Cross-Attention Weights')
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    
    plt.tight_layout()
    plt.show()
```

## Modern Transformer Modelleri

### 1. BERT (Bidirectional Encoder Representations from Transformers)
```python
# BERT ile metin sınıflandırma
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessed = bert_preprocess(text_input)
bert_outputs = bert_encoder(preprocessed)

# Sınıflandırma katmanı
dropout = tf.keras.layers.Dropout(0.1)(bert_outputs['pooled_output'])
output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)

model = tf.keras.Model(inputs=text_input, outputs=output)
```

### 2. GPT (Generative Pre-trained Transformer)
```python
# GPT benzeri dil modeli
class GPTModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, max_length):
        super(GPTModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff)
            for _ in range(num_layers)
        ]
```

## 📚 Önerilen Kaynaklar
- [Attention is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. BERT ile metin sınıflandırma
2. Basit bir Transformer encoder implementasyonu

### Orta Seviye
1. Makine çevirisi uygulaması
2. Custom attention layer oluşturma

### İleri Seviye
1. Pre-training ve fine-tuning
2. Multi-task Transformer modeli geliştirme 