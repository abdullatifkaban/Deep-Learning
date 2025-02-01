# Model Optimizasyonu

## 📍 Bölüm Haritası
- Önceki Bölüm: [Reinforcement Learning](03-Reinforcement-Learning.md)
- Sonraki Bölüm: [Model-Deployment](../06-Deployment/01-Model-Deployment.md)
- Tahmini Süre: 5-6 saat
- Zorluk Seviyesi: 🔴 İleri

## 🎯 Hedefler
- Model optimizasyon tekniklerini anlama
- Hyperparameter tuning yapabilme
- Model compression uygulama
- Inference hızını artırma

## 🎯 Öz Değerlendirme
- [ ] Optimizasyon tekniklerini açıklayabiliyorum
- [ ] Hyperparameter tuning yapabiliyorum
- [ ] Model compression uygulayabiliyorum
- [ ] Inference performansını artırabiliyorum

## 🚀 Mini Projeler
1. Hyperparameter Tuning
   - Grid/Random search
   - Bayesian optimization
   - Cross validation

2. Model Compression
   - Quantization
   - Pruning
   - Knowledge distillation

## 📑 Ön Koşullar
- Derin öğrenme temelleri
- Python ve framework deneyimi
- Optimizasyon teorisi
- Model mimarileri bilgisi

## 🔑 Temel Kavramlar
1. Hyperparameter Tuning
2. Model Compression
3. Quantization
4. Knowledge Distillation

## Giriş

Model optimizasyonu, derin öğrenme modellerinin performansını ve verimliliğini artırmak için kullanılan teknikleri içerir.

## Hiperparametre Optimizasyonu

### 1. Grid Search
```python
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_model(learning_rate=0.01, neurons=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(neurons//2, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Grid Search parametreleri
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'neurons': [32, 64, 128],
    'batch_size': [32, 64, 128],
    'epochs': [10, 50, 100]
}

model = KerasClassifier(build_fn=create_model)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)
```

### 2. Random Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Random Search parametreleri
param_dist = {
    'learning_rate': uniform(0.0001, 0.1),
    'neurons': randint(32, 512),
    'batch_size': randint(16, 128),
    'epochs': randint(10, 100)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    cv=3
)
random_search.fit(X_train, y_train)
```

## Model Pruning

### 1. Magnitude-based Pruning
```python
import tensorflow_model_optimization as tfmot

# Pruning konfigürasyonu
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

# Model pruning
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, **pruning_params)

# Pruning callback
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
```

### 2. Channel Pruning
```python
def channel_pruning(model, pruning_ratio):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()[0]
            importance = np.sum(np.abs(weights), axis=(0,1,2))
            threshold = np.percentile(importance, pruning_ratio * 100)
            mask = importance > threshold
            weights[:,:,:,~mask] = 0
            layer.set_weights([weights])
```

## Quantization

### 1. Post-training Quantization
```python
# TFLite dönüşümü ve quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
```

### 2. Quantization-aware Training
```python
# Quantization-aware model oluşturma
quantize_model = tfmot.quantization.keras.quantize_model

# Apply quantization to the model
q_aware_model = quantize_model(model)

# Model derleme
q_aware_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Quantization-aware training
q_aware_model.fit(train_data,
                 epochs=epochs,
                 validation_data=val_data)
```

## Model Distillation

### 1. Knowledge Distillation
```python
class DistillationModel(tf.keras.Model):
    def __init__(self, student, teacher):
        super(DistillationModel, self).__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = 3.0
        
    def compile(self, optimizer, metrics):
        super(DistillationModel, self).compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()
        
    def train_step(self, data):
        x, y = data
        
        # Teacher tahminleri
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Student tahminleri
            student_predictions = self.student(x, training=True)
            
            # Distillation loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature),
                tf.nn.softmax(student_predictions / self.temperature)
            )
            
            # Student loss
            student_loss = self.compiled_loss(y, student_predictions)
            
            # Toplam loss
            total_loss = (0.9 * distillation_loss) + (0.1 * student_loss)
        
        # Gradyan hesaplama ve uygulama
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {"loss": total_loss, "student_loss": student_loss, "distillation_loss": distillation_loss}
```

## Model Compression

### 1. Weight Pruning ve Compression
```python
def compress_weights(model, compression_ratio=0.9):
    compressed_weights = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()[0]
            threshold = np.percentile(np.abs(weights), compression_ratio * 100)
            mask = np.abs(weights) > threshold
            sparse_weights = weights * mask
            compressed_weights.append(sparse_weights)
    return compressed_weights
```

### 2. Layer Fusion
```python
def fuse_batch_norm(model):
    for i in range(len(model.layers) - 1):
        if (isinstance(model.layers[i], tf.keras.layers.Conv2D) and
            isinstance(model.layers[i+1], tf.keras.layers.BatchNormalization)):
            
            conv = model.layers[i]
            bn = model.layers[i+1]
            
            # Batch norm parametreleri
            gamma = bn.gamma
            beta = bn.beta
            mean = bn.moving_mean
            var = bn.moving_variance
            
            # Yeni konvolüsyon ağırlıkları
            w = conv.kernel
            if conv.use_bias:
                b = conv.bias
            else:
                b = tf.zeros_like(beta)
            
            # Fusion hesaplama
            w_new = w * gamma / tf.sqrt(var + bn.epsilon)
            b_new = beta + gamma * (b - mean) / tf.sqrt(var + bn.epsilon)
            
            # Yeni ağırlıkları ayarla
            conv.kernel = w_new
            conv.bias = b_new
```

## Alıştırmalar

1. Hiperparametre Optimizasyonu:
   - Kendi modeliniz için grid search uygulayın
   - Random search ile karşılaştırın
   - En iyi parametreleri bulun

2. Model Pruning:
   - CNN modelinde pruning uygulayın
   - Farklı pruning oranlarını deneyin
   - Performans/boyut trade-off'unu analiz edin

3. Quantization:
   - Modelinizi TFLite'a dönüştürün
   - Farklı quantization stratejilerini deneyin
   - Mobil cihazda test edin

## Kaynaklar
1. [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
2. [Knowledge Distillation Tutorial](https://www.tensorflow.org/tutorials/optimization/knowledge_distillation)
3. [TFLite Model Optimization Guide](https://www.tensorflow.org/lite/performance/model_optimization) 