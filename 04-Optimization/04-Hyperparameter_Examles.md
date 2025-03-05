### 7. Hiperparametre Uygulama Örnekleri

#### 7.1 Temel Model Oluşturma

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Veri setini yükleme
# Veri setini yükleme
data = pd.read_csv('pima-indians-diabetes.csv')

X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Veri ön işleme
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Temel model
def create_base_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification için sigmoid
    ])
    return model
```

#### 7.2 Öğrenme Oranı Optimizasyonu
```python
learning_rates = [0.1, 0.01, 0.001, 0.0001]
lr_histories = {}
lr_accuracies = {}

for lr in learning_rates:
    model = create_base_model()
    model.compile(optimizer=Adam(learning_rate=lr),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,
                       validation_split=0.2,
                       epochs=50,
                       batch_size=32,
                       verbose=0)
    lr_histories[lr] = history.history
    
    # Sınıflandırma performansı değerlendirme
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()
    lr_accuracies[lr] = (y_pred == y_test.values).mean()
```
**Sonuçları Görselleştirme**
```python
plt.figure(figsize=(15, 4))

# Eğitim doğruluğu (Accuracy)
plt.subplot(1, 3, 1)
for lr in learning_rates:
    plt.plot(lr_histories[lr]['accuracy'], label=f'LR={lr}')
plt.title('Eğitim Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Doğrulama doğruluğu
plt.subplot(1, 3, 2)
for lr in learning_rates:
    plt.plot(lr_histories[lr]['val_accuracy'], label=f'LR={lr}')
plt.title('Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()

# Test set doğruluk skorları
plt.subplot(1, 3, 3)
plt.plot(learning_rates, [lr_accuracies[lr] for lr in learning_rates], marker='o')
plt.title('Learning Rate vs Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.xscale('log')

plt.tight_layout()
plt.show()
```

#### 7.3 Batch Size Optimizasyonu

```python
batch_sizes = [16, 32, 64, 128]
batch_histories = {}
batch_accuracies = {}

for batch_size in batch_sizes:
    model = create_base_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',  # Sınıflandırma problemi için binary_crossentropy
                 metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,
                       validation_split=0.2,
                       epochs=50,
                       batch_size=batch_size,
                       verbose=0)
    batch_histories[batch_size] = history.history
    
    # Accuracy hesaplama
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()
    batch_accuracies[batch_size] = accuracy_score(y_test, y_pred)
```
**Sonuçları Görselleştirme**

```python
plt.figure(figsize=(15, 4))

# Eğitim doğruluğu (Accuracy)
plt.subplot(1, 3, 1)
for batch_size in batch_sizes:
    plt.plot(batch_histories[batch_size]['accuracy'], label=f'Batch Size={batch_size}')
plt.title('Eğitim Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Doğrulama doğruluğu
plt.subplot(1, 3, 2)
for batch_size in batch_sizes:
    plt.plot(batch_histories[batch_size]['val_accuracy'], label=f'Batch Size={batch_size}')
plt.title('Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()

# Test set doğruluk skorları
plt.subplot(1, 3, 3)
plt.plot(batch_sizes, [batch_accuracies[batch_size] for batch_size in batch_sizes], marker='o')
plt.title('Batch Size vs Accuracy')
plt.xlabel('Batch Size')
plt.ylabel('Test Accuracy')
plt.xscale('log')

plt.tight_layout()
plt.show()
```

#### 7.4 Early Stopping Uygulaması

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model = create_base_model()
model.compile(optimizer=Adam(learning_rate=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])

history_es = model.fit(X_train, y_train,
                      validation_split=0.2,
                      epochs=100,
                      batch_size=32,
                      callbacks=[early_stopping],
                      verbose=1)
```
**Sonuçları Görselleştirme**

```python
plt.figure(figsize=(15, 6))

# Eğitim ve doğrulama kayıpları
plt.subplot(1, 2, 1)
plt.plot(history_es.history['loss'], label='Eğitim Kaybı')
plt.plot(history_es.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kaybı')
plt.legend()

# Eğitim ve doğrulama doğruluğu
plt.subplot(1, 2, 2)
plt.plot(history_es.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_es.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.tight_layout()
plt.show()
```
#### 7.5 Öğrenme Oranı Planlaması

```python
def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * (drop ** np.floor((1+epoch)/epochs_drop))
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)

model = create_base_model()
model.compile(optimizer=Adam(learning_rate=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])

history_lrs = model.fit(X_train, y_train,
                       validation_split=0.2,
                       epochs=50,
                       batch_size=32,
                       callbacks=[lr_scheduler],
                       verbose=1)
```

**Sonuçların Görselleştirilmesi**

```python
# Eğitim geçmişinden kayıpları ve doğrulukları al
history = history_lrs.history

# Kayıpları çizme
plt.figure(figsize=(12, 5))

# Kaybı çiz
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Eğitim Kaybı')
plt.plot(history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kaybın Zamanla Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid()

# Doğruluğu çiz
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluğun Zamanla Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid()

# Grafiği göster
plt.tight_layout()
plt.show()
```
#### 7.6 Grid Search

```python
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Model oluşturucu fonksiyon
def create_model(neurons=64, learning_rate=0.001, dropout_rate=0.2):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(neurons//2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# KerasClassifier modeli oluştur
model = KerasClassifier(model=create_model, 
                        epochs=50, 
                        batch_size=32, 
                        verbose=0)

# Grid Search parametreleri
param_grid = {
    'model__neurons': [32, 64, 128],  # 'model__' prefixi kullanarak parametreleri belirtin
    'model__learning_rate': [0.1, 0.01, 0.001],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'batch_size': [16, 32, 64]
}

# Grid Search uygula
grid = GridSearchCV(estimator=model, 
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    verbose=2)

grid_result = grid.fit(X_train, y_train)

# En iyi parametreleri ve skoru yazdır
print(f"En iyi parametreler: {grid_result.best_params_}")
print(f"En iyi çapraz doğrulama skoru: {grid_result.best_score_:.4f}")

# Test seti üzerinde değerlendir
test_score = grid_result.score(X_test, y_test)
print(f"Test seti doğruluk skoru: {test_score:.4f}")
```
#### 7.6 Random Search

```Python
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np

# Model oluşturucu fonksiyon
def create_model(neurons=64, learning_rate=0.001, dropout_rate=0.2):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(neurons//2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# KerasClassifier modeli oluştur
model = KerasClassifier(model=create_model, 
                        epochs=50, 
                        batch_size=32, 
                        verbose=0)

# Random Search parametreleri
param_distributions = {
    'model__neurons': [32, 64, 128],  # 'model__' prefixi kullanarak parametreleri belirtin
    'model__learning_rate': [0.1, 0.01, 0.001],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'batch_size': [16, 32, 64]
}

# Random Search uygula
random_search = RandomizedSearchCV(estimator=model, 
                                   param_distributions=param_distributions,
                                   n_iter=10,  # Denenecek rastgele kombinasyon sayısı
                                   cv=3,
                                   n_jobs=-1,
                                   verbose=2,
                                   random_state=42)

random_result = random_search.fit(X_train, y_train)
```
**Sonuçların Görselleştirilmesi**
```python
import matplotlib.pyplot as plt
import pandas as pd

# En iyi sonuçları elde et
results = random_search.cv_results_

# Sonuçları DataFrame'e çevir
results_df = pd.DataFrame(results)

# En iyi sonuçları ve parametreleri seç
best_results = results_df[['mean_test_score', 'std_test_score', 'params']]

# Sonuçları sıralama
best_results = best_results.sort_values(by='mean_test_score', ascending=False)

# Grafik oluşturma
plt.figure(figsize=(12, 6))
plt.barh(best_results.index, best_results['mean_test_score'], xerr=best_results['std_test_score'], color='skyblue')
plt.yticks(best_results.index, best_results['params'].apply(lambda x: str(x)))
plt.xlabel('Mean Test Score (Accuracy)')
plt.title('Randomized Search Results')
plt.grid(axis='x')

# Grafik gösterme
plt.tight_layout()
plt.show()
```