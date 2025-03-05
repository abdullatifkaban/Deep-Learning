# Hiperparametre Ayarlamaları

Derin öğrenme modellerinin başarısı, doğru mimariyi seçmenin yanı sıra eğitim sürecinde kullanılan hiperparametrelerin (öğrenme oranı, batch size, epoch sayısı vb.) doğru ayarlanmasına da bağlıdır. Bu belgede, hiperparametre ayarlamalarının önemini, her birinin etkisini ve pratik Python örnekleriyle nasıl ayarlandığını inceleyeceğiz.

---

## 1. Giriş

Hiperparametreler, model eğitimi başlamadan önce ayarlanan ve eğitim sürecinin davranışını etkileyen değerlerdir. Doğru hiperparametre ayarları, modelin daha hızlı öğrenmesine, aşırı öğrenme (overfitting) veya yetersiz öğrenme (underfitting) gibi problemlerden kaçınmasına yardımcı olur. Temel hiperparametreler arasında öğrenme oranı, batch size ve epoch sayısı yer alır.

---

## 2. Öğrenme Oranı (Learning Rate)

**Tanım:** Öğrenme oranı, modelin ağırlıklarını güncellerken kullanılan adım büyüklüğünü belirler. Çok yüksek bir öğrenme oranı, modelin optimum değerden sapmasına neden olabilirken, çok düşük bir öğrenme oranı ise modelin yavaş öğrenmesine ve yerel minimuma takılmasına yol açabilir.

**Etkileri:**

* **Yüksek Öğrenme Oranı:** Hızlı öğrenme, ancak kararsızlık riski.
* **Düşük Öğrenme Oranı:** Daha istikrarlı öğrenme, ancak eğitim süresi uzar.

**Kod Örneği:**

```python
from tensorflow.keras.optimizers import Adam

# Farklı öğrenme oranları ile model derleme
optimizer_high_lr = Adam(learning_rate=0.1)
optimizer_low_lr = Adam(learning_rate=0.001)

# Model derlemesinde kullanılacak öğrenme oranı, deneysel olarak ayarlanır.
model.compile(optimizer=optimizer_low_lr, loss='binary_crossentropy', metrics=['accuracy'])
```

**Not:** Genellikle başlangıç için 0.001 değeri tercih edilir. Modelin eğitim kaybı ve doğruluk grafikleri takip edilerek öğrenme oranı ayarlanabilir.


## 3. Batch Size

**Tanım:** Batch size, modelin eğitim verisinden her seferinde kaç örneği işlediğini belirler. Tüm eğitim verisinin tek seferde işlenmesi (batch size = tüm veri) çok yüksek bellek kullanımı gerektirirken, çok küçük batch size değerleri daha gürültülü gradyanlar üretebilir.

**Etkileri:**

* **Büyük Batch Size:** Daha stabil gradyanlar, ancak daha yüksek bellek gereksinimi ve bazen genel genelleme performansında azalma.
* **Küçük Batch Size:** Daha sık ağırlık güncellemeleri, potansiyel olarak daha iyi genelleme, ancak eğitim sırasında gürültü artar.

**Kod Örneği:**

```python
# Modeli eğitirken kullanılacak batch size değeri
batch_size = 16

history = model.fit(X, y, epochs=100, batch_size=batch_size, verbose=1)
```

**Not:** Batch size, veri seti büyüklüğüne ve kullanılan donanıma göre ayarlanmalıdır.


## 4. Epoch Sayısı

**Tanım:** Epoch, modelin tüm eğitim verisi üzerinden bir kez geçmesi anlamına gelir. Epoch sayısı, modelin ne kadar süre eğitileceğini belirler.

**Etkileri:**

* **Yüksek Epoch Sayısı:** Model daha uzun süre eğitilir, ancak aşırı öğrenme riski artar.
* **Düşük Epoch Sayısı:** Eğitim süresi kısalır, fakat model yeterince öğrenemeyebilir.

**Kod Örneği:**

```python
# Model eğitimi sırasında kullanılacak epoch sayısı
epochs = 100

history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
```

**Not:** Eğitim sırasında kayıp ve doğruluk değerlerinin izlenmesi, epoch sayısının yeterliliğini anlamak için önemlidir.


## 5. Hiperparametre Ayarlamalarında Stratejiler

### Grid Search

Grid Search, belirtilen hiperparametre aralıklarındaki tüm olası kombinasyonları sistematik olarak deneyen bir yöntemdir. Her kombinasyon test edildiği için en iyi sonucu garanti eder, ancak hesaplama maliyeti yüksektir.

```python
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(learning_rate=0.001):
    model = Sequential([
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

model = KerasClassifier(build_fn=create_model)
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X, y)
print(f"En iyi parametreler: {grid_result.best_params_}")
```

### Random Search

Random Search, hiperparametre uzayından rastgele kombinasyonlar seçerek daha hızlı sonuç elde etmeyi amaçlar. Grid Search kadar kapsamlı değildir ancak genellikle daha verimlidir.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'learning_rate': uniform(0.0001, 0.1),
    'batch_size': randint(16, 128),
    'epochs': randint(30, 100)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3
)

random_result = random_search.fit(X, y)
print(f"En iyi parametreler: {random_result.best_params_}")
```

### Early Stopping Uygulaması

Validation loss'un artmaya başladığı noktada eğitimi durdurmak için:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stopping]
)
```

### Learning Rate Scheduler

Eğitim sırasında öğrenme oranını dinamik olarak ayarlamak için:

```python
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

model.fit(X, y, epochs=100, callbacks=[lr_scheduler])
```

### Batch Size Optimizasyonu

Farklı batch size'ları test etmek için:

```python
batch_sizes = [16, 32, 64]
histories = {}

for batch_size in batch_sizes:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size
    )
    histories[batch_size] = history.history
```

Bu stratejiler, modelin performansını optimize etmek için sistematik bir yaklaşım sunar. Her veri seti ve model mimarisi için en uygun değerleri bulmak önemlidir.


## 6. Sonuç

Hiperparametre ayarlamaları, derin öğrenme modellerinin performansını doğrudan etkileyen kritik adımlardandır. Öğrenme oranı, batch size ve epoch sayısının dikkatli bir şekilde seçilmesi ve deneysel olarak optimize edilmesi, modelin daha hızlı ve istikrarlı bir şekilde öğrenmesine olanak tanır. Bu temel ayarlamaların yanı sıra, ileri seviye yöntemlerle (örneğin, learning rate scheduler, early stopping) eğitim sürecini daha verimli hale getirmek mümkündür. Her model ve veri seti için en iyi hiperparametre kombinasyonunu bulmak deneysel bir süreçtir. Bu nedenle, eğitim sırasında modelin performansını sürekli izlemek ve ayarlamalara esnek yaklaşmak önemlidir.