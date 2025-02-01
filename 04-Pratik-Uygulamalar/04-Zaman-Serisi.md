# Zaman Serisi Analizi

## 🎯 Hedefler
- Zaman serisi verilerini anlama ve ön işleme
- Farklı tahmin modellerini uygulama
- Uzun ve kısa vadeli tahminler yapabilme
- Model performansını değerlendirme ve iyileştirme

## 📑 Ön Koşullar
- RNN ve LSTM mimarileri bilgisi
- Python ve TensorFlow/PyTorch deneyimi
- Temel istatistik ve olasılık
- Veri analizi ve görselleştirme becerileri

## 🔑 Temel Kavramlar
1. Zaman Serisi Bileşenleri
2. Mevsimsellik ve Trend
3. Durağanlık
4. Özellik Mühendisliği
5. Tahmin Metrikleri

## Veri Hazırlama
> Zorluk Seviyesi: 🟡 Orta

💡 İpucu: Doğru veri ön işleme ve özellik mühendisliği, model başarısını önemli ölçüde etkiler

### 1. Veri Yükleme ve Ön İşleme
```python
def prepare_time_series_data(data, sequence_length=10):
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(targets)
```

### 2. Zaman Serisi Verisi Oluşturma
```python
import tensorflow as tf
import numpy as np
import pandas as pd

def create_time_series_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Örnek veri
time = np.arange(0, 100, 0.1)
data = np.sin(time) + np.random.normal(0, 0.1, len(time))

# Veri seti oluşturma
X, y = create_time_series_dataset(data, time_steps=50)
```

### 3. Veri Normalizasyonu
```python
# Min-Max normalizasyon
class MinMaxNormalizer:
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, data):
        self.min = np.min(data)
        self.max = np.max(data)
    
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)
    
    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

# Veriyi normalize et
normalizer = MinMaxNormalizer()
normalizer.fit(data)
X_norm = normalizer.transform(X)
y_norm = normalizer.transform(y)
```

## Model Oluşturma

### 1. LSTM Modeli
```python
def build_lstm_model(time_steps, features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                           input_shape=(time_steps, features)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model
```

### 2. CNN-LSTM Modeli
```python
def build_cnn_lstm_model(time_steps, features):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                             activation='relu',
                             input_shape=(time_steps, features)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    return model
```

## Model Eğitimi

### 1. Eğitim Konfigürasyonu
```python
# Model derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='mse',
             metrics=['mae'])

# Callback'ler
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]
```

### 2. Eğitim ve Değerlendirme
```python
# Model eğitimi
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Eğitim görselleştirme
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.legend()
plt.show()
```

## Tahmin ve Değerlendirme

### 1. Tek Adım Tahmin
```python
def single_step_prediction(model, X):
    prediction = model.predict(X)
    return normalizer.inverse_transform(prediction)

# Test verisi üzerinde tahmin
y_pred = single_step_prediction(model, X_test)
```

### 2. Çok Adımlı Tahmin
```python
def multi_step_prediction(model, initial_sequence, steps):
    current_sequence = initial_sequence.copy()
    predictions = []
    
    for _ in range(steps):
        # Tek adım tahmin
        next_pred = model.predict(current_sequence[np.newaxis, :, :])
        predictions.append(next_pred[0, 0])
        
        # Sequence'i güncelle
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    return np.array(predictions)

# Gelecek 50 adım için tahmin
future_pred = multi_step_prediction(model, X_test[-1], steps=50)
```

## Özel Uygulamalar

### 1. Hisse Senedi Fiyat Tahmini
```python
def prepare_stock_data(df, feature_columns, target_column):
    # Özellik seçimi
    features = df[feature_columns]
    target = df[target_column]
    
    # Veri normalizasyonu
    scaler = MinMaxNormalizer()
    features_normalized = scaler.fit_transform(features)
    
    # Zaman serisi veri seti oluşturma
    X, y = create_time_series_dataset(features_normalized, time_steps=30)
    
    return X, y, scaler

# Model oluşturma ve eğitim
stock_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
```

### 2. Hava Durumu Tahmini
```python
def build_weather_model(num_features):
    inputs = tf.keras.layers.Input(shape=(None, num_features))
    
    # CNN katmanları
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM katmanları
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(32)(x)
    
    # Yoğun katmanlar
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

## Örnek Uygulamalar

### 3. Çok Değişkenli Tahmin
```python
def build_multivariate_forecaster():
    # Giriş katmanı
    inputs = tf.keras.layers.Input(shape=(lookback, n_features))
    
    # Çift yönlü LSTM katmanları
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Attention mekanizması
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(128)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    
    # Attention ve LSTM çıktılarını birleştir
    sent_representation = tf.keras.layers.multiply([x, attention])
    sent_representation = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1))(sent_representation)
    
    # Tahmin katmanları
    x = tf.keras.layers.Dense(64, activation='relu')(sent_representation)
    outputs = tf.keras.layers.Dense(n_features * forecast_horizon)(x)
    outputs = tf.keras.layers.Reshape((forecast_horizon, n_features))(outputs)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### 4. Anomali Tespiti
```python
class TimeSeriesAnomalyDetector:
    def __init__(self, sequence_length=100, threshold=0.95):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            # Encoder
            tf.keras.layers.LSTM(64, activation='tanh', input_shape=(self.sequence_length, 1), 
                               return_sequences=True),
            tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True),
            tf.keras.layers.LSTM(16, activation='tanh', return_sequences=False),
            
            # Decoder
            tf.keras.layers.RepeatVector(self.sequence_length),
            tf.keras.layers.LSTM(16, activation='tanh', return_sequences=True),
            tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True),
            tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def detect_anomalies(self, data):
        # Rekonstrüksiyon hatası
        reconstructed = self.model.predict(data)
        mse = np.mean(np.square(data - reconstructed), axis=1)
        
        # Anomali skoru
        threshold = np.percentile(mse, self.threshold * 100)
        anomalies = mse > threshold
        
        return anomalies, mse
```

### 5. Mevsimsel Ayrıştırma
```python
class SeasonalDecomposer:
    def __init__(self, period=None):
        self.period = period
        self.trend = None
        self.seasonal = None
        self.residual = None
    
    def decompose(self, data, type='multiplicative'):
        # Trend bileşeni
        def extract_trend(data, window):
            return pd.Series(data).rolling(window=window, center=True).mean()
        
        # Mevsimsel bileşen
        def extract_seasonal(data, period):
            seasonal_means = pd.Series(data).groupby(pd.Series(range(len(data))) % period).mean()
            return np.tile(seasonal_means, len(data)//period + 1)[:len(data)]
        
        if self.period is None:
            # Otomatik periyot tespiti
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(data, nlags=len(data)//2)
            peaks = signal.find_peaks(acf_vals)[0]
            self.period = peaks[0] if len(peaks) > 0 else 1
        
        if type == 'multiplicative':
            self.trend = extract_trend(data, self.period)
            detrended = data / self.trend
            self.seasonal = extract_seasonal(detrended, self.period)
            self.residual = detrended / self.seasonal
        else:  # additive
            self.trend = extract_trend(data, self.period)
            detrended = data - self.trend
            self.seasonal = extract_seasonal(detrended, self.period)
            self.residual = detrended - self.seasonal
        
        return self.trend, self.seasonal, self.residual
```

### 6. Hibrit Tahmin Modeli
```python
class HybridForecaster:
    def __init__(self, lookback=30, horizon=10):
        self.lookback = lookback
        self.horizon = horizon
        self.statistical_model = None  # SARIMA için
        self.deep_model = self._build_deep_model()
    
    def _build_deep_model(self):
        model = tf.keras.Sequential([
            # CNN katmanları - kısa vadeli pattern'ler için
            tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(self.lookback, 1)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            
            # LSTM katmanları - uzun vadeli bağımlılıklar için
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            
            # Tahmin katmanları
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.horizon)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, data):
        # SARIMA model fitting
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.statistical_model = SARIMAX(data, order=(1,1,1), 
                                       seasonal_order=(1,1,1,12))
        self.statistical_model = self.statistical_model.fit(disp=False)
        
        # Residuals için deep learning
        residuals = data - self.statistical_model.fittedvalues
        X, y = self._prepare_sequences(residuals)
        self.deep_model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    
    def predict(self, data):
        # SARIMA tahmini
        stat_pred = self.statistical_model.forecast(self.horizon)
        
        # Deep learning residual tahmini
        X = self._prepare_sequences(data)
        deep_pred = self.deep_model.predict(X)
        
        # Tahminleri birleştir
        return stat_pred + deep_pred
```

## 📚 Önerilen Kaynaklar
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Time Series Analysis with Python](https://machinelearningmastery.com/time-series-forecasting-with-python/)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. Basit LSTM ile hava durumu tahmini
2. Temel zaman serisi analizi ve görselleştirme

### Orta Seviye
1. Çok değişkenli zaman serisi tahmini
2. Attention mekanizması ekleme

### İleri Seviye
1. Hibrit tahmin modeli geliştirme
2. Gerçek zamanlı anomali tespiti sistemi 