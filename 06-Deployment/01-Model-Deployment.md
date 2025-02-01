# Model Deployment ve Servisleştirme

## 🎯 Hedefler
- Model deployment sürecini anlama
- REST API servisleri oluşturma
- Docker containerization uygulama
- Model versiyonlama ve yönetimi

## 📑 Ön Koşullar
- Python web framework'leri (Flask/FastAPI)
- Docker temel bilgisi
- RESTful API kavramları
- Temel Linux komutları

## 🔑 Temel Kavramlar
1. Model Servisleştirme
2. API Tasarımı
3. Containerization
4. Load Balancing
5. Model Versiyonlama

## 📍 Bölüm Haritası
- Önceki Bölüm: [Zaman Serisi Analizi](../04-Pratik-Uygulamalar/04-Zaman-Serisi.md)
- Sonraki Bölüm: [Model Monitoring](02-Model-Monitoring.md)
- Tahmini Süre: 6-7 saat
- Zorluk Seviyesi: 🔴 İleri

## 🎯 Öz Değerlendirme
- [ ] REST API servisleri oluşturabiliyorum
- [ ] Docker containerization yapabiliyorum
- [ ] Model versiyonlama uygulayabiliyorum
- [ ] Ölçeklendirme stratejileri geliştirebiliyorum

## 🚀 Mini Projeler
1. Model Servis API'si
   - FastAPI ile REST servisi geliştirin
   - Docker container oluşturun
   - Load balancing ekleyin

2. MLOps Pipeline
   - CI/CD pipeline kurun
   - Model registry oluşturun
   - Monitoring sistemi entegre edin

## Model Servisleştirme
> Zorluk Seviyesi: 🔴 İleri

> 💡 İpucu: Modeli servisleştirmeden önce performans optimizasyonu yapın

### 1. FastAPI ile REST API
```python
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('model.h5')

@app.post("/predict")
async def predict(data: dict):
    # Veriyi ön işle
    processed_data = preprocess_data(data)
    
    # Tahmin yap
    predictions = model.predict(processed_data)
    
    return {"predictions": predictions.tolist()}
```

### 2. Model Kaydetme
```python
import tensorflow as tf

# SavedModel formatında kaydetme
model.save('saved_model/1/')

# Model versiyonlama
model.save('saved_model/{version}/')
```

### 3. Docker ile TF Serving
```bash
# TensorFlow Serving Docker imajını çek
docker pull tensorflow/serving

# Modeli serve et
docker run -p 8501:8501 \
    --mount type=bind,source=/path/to/saved_model,target=/models/mymodel \
    -e MODEL_NAME=mymodel \
    tensorflow/serving
```

### 4. REST API ile İstek
```python
import requests
import json

def predict(data):
    headers = {"content-type": "application/json"}
    json_data = json.dumps({
        "instances": data.tolist()
    })
    
    json_response = requests.post(
        'http://localhost:8501/v1/models/mymodel:predict',
        data=json_data,
        headers=headers
    )
    predictions = json.loads(json_response.text)
    return predictions
```

## FastAPI ile Web Service

### 1. FastAPI Uygulaması
```python
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('saved_model/1')

class PredictionInput(BaseModel):
    data: list

@app.post("/predict")
def predict(input_data: PredictionInput):
    data = tf.convert_to_tensor(input_data.data)
    predictions = model(data)
    return {"predictions": predictions.numpy().tolist()}
```

### 2. Middleware ve Güvenlik
```python
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
    )
```

## Docker Containerization

### 1. Dockerfile
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose
```yaml
version: '3'
services:
  model-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.h5
      - API_KEY=${API_KEY}
```

## Model Versiyonlama

### 1. MLflow ile Model Tracking
```python
import mlflow
import mlflow.tensorflow

mlflow.tensorflow.autolog()

with mlflow.start_run():
    # Model eğitimi
    history = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data
    )
    
    # Model kaydetme
    mlflow.tensorflow.log_model(model, "model")
```

### 2. Model Registry
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Model versiyonu oluştur
model_version = client.create_model_version(
    name="my_model",
    source="mlruns/0/run_id/artifacts/model",
    run_id="run_id"
)

# Staging'e geçir
client.transition_model_version_stage(
    name="my_model",
    version=model_version.version,
    stage="Staging"
)
```

## A/B Testing

### 1. Test Konfigürasyonu
```python
class ABTest:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.metrics = {'a': [], 'b': []}
    
    def predict(self, data):
        if np.random.random() < self.split_ratio:
            pred = self.model_a.predict(data)
            self.metrics['a'].append(pred)
            return {'model': 'A', 'prediction': pred}
        else:
            pred = self.model_b.predict(data)
            self.metrics['b'].append(pred)
            return {'model': 'B', 'prediction': pred}
```

### 2. Performans İzleme
```python
def monitor_performance(metrics_a, metrics_b):
    from scipy import stats
    
    # İstatistiksel test
    t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b)
    
    return {
        'model_a_mean': np.mean(metrics_a),
        'model_b_mean': np.mean(metrics_b),
        't_statistic': t_stat,
        'p_value': p_value
    }
```

## Otomatik Yeniden Eğitim

### 1. Veri Drift Tespiti
```python
from alibi_detect.cd import KSDrift

def detect_drift(reference_data, new_data, threshold=0.05):
    drift_detector = KSDrift(
        reference_data,
        p_val=threshold
    )
    
    drift_prediction = drift_detector.predict(new_data)
    return drift_prediction['data']['is_drift']
```

### 2. Otomatik Eğitim Pipeline
```python
class AutoTrainingPipeline:
    def __init__(self, model_builder, data_loader):
        self.model_builder = model_builder
        self.data_loader = data_loader
        
    def train_if_needed(self):
        new_data = self.data_loader.get_new_data()
        if self.should_retrain(new_data):
            model = self.model_builder()
            model.fit(new_data)
            self.save_and_deploy(model)
    
    def should_retrain(self, new_data):
        return detect_drift(self.reference_data, new_data)
```

## Örnek Uygulamalar

### 3. Model Versiyonlama ve A/B Testing
```python
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.active_model = None
        self.experiment_model = None
        self.traffic_split = 0.0
    
    def register_model(self, model_id, model, is_experiment=False):
        self.models[model_id] = model
        if is_experiment:
            self.experiment_model = model_id
        else:
            self.active_model = model_id
    
    def set_traffic_split(self, split_ratio):
        self.traffic_split = max(0.0, min(1.0, split_ratio))
    
    def get_model(self, request_id):
        # Request ID'ye göre tutarlı model seçimi
        if self.experiment_model and hash(request_id) % 100 < self.traffic_split * 100:
            return self.models[self.experiment_model]
        return self.models[self.active_model]
```

### 4. Model Monitoring ve Logging
```python
class ModelMonitor:
    def __init__(self):
        self.metrics = {}
        self.predictions = []
        self.latencies = []
        
    def log_prediction(self, input_data, prediction, latency, metadata=None):
        log_entry = {
            'timestamp': datetime.now(),
            'input': input_data,
            'prediction': prediction,
            'latency': latency,
            'metadata': metadata
        }
        
        # Prometheus metriklerini güncelle
        self.update_metrics(log_entry)
        
        # Tahminleri kaydet
        self.predictions.append(log_entry)
        
        # Performans metriklerini güncelle
        self.latencies.append(latency)
        
        # Drift tespiti
        if len(self.predictions) % 1000 == 0:
            self.check_drift()
    
    def update_metrics(self, log_entry):
        # Prometheus metriklerini güncelle
        PREDICTION_COUNTER.inc()
        LATENCY_HISTOGRAM.observe(log_entry['latency'])
        
    def check_drift(self):
        recent_data = self.predictions[-1000:]
        # Drift analizi
        drift_score = self.calculate_drift(recent_data)
        if drift_score > DRIFT_THRESHOLD:
            self.alert_drift()
```

### 5. Model Optimizasyonu
```python
def optimize_model_for_deployment(model):
    # TensorRT optimizasyonu
    params = tf.experimental.tensorrt.ConversionParams(
        precision_mode='FP16',
        maximum_cached_engines=1000
    )
    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=model_path,
        conversion_params=params
    )
    
    # Modeli optimize et
    converter.convert()
    
    # Quantization uygula
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Modeli kaydet
    tflite_model = converter.convert()
    return tflite_model
```

### 6. Batch Prediction Service
```python
class BatchPredictionService:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.prediction_queue = Queue()
        self.result_dict = {}
        self.lock = threading.Lock()
        
    def start_processing(self):
        self.processing_thread = threading.Thread(target=self._process_batches)
        self.processing_thread.start()
        
    def _process_batches(self):
        while True:
            batch = []
            batch_ids = []
            
            # Batch oluştur
            while len(batch) < self.batch_size:
                try:
                    request_id, data = self.prediction_queue.get(timeout=0.1)
                    batch.append(data)
                    batch_ids.append(request_id)
                except Empty:
                    if batch:  # Mevcut batch'i işle
                        break
                    continue
            
            if not batch:
                continue
                
            # Batch tahminini yap
            predictions = self.model.predict(np.array(batch))
            
            # Sonuçları kaydet
            with self.lock:
                for req_id, pred in zip(batch_ids, predictions):
                    self.result_dict[req_id] = pred
    
    def predict(self, data):
        request_id = str(uuid.uuid4())
        
        # Tahmin isteğini kuyruğa ekle
        self.prediction_queue.put((request_id, data))
        
        # Sonucu bekle
        while True:
            with self.lock:
                if request_id in self.result_dict:
                    return self.result_dict.pop(request_id)
            time.sleep(0.01)
```

## 📚 Önerilen Kaynaklar
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. FastAPI ile basit model servisi
2. Docker container oluşturma

### Orta Seviye
1. Model versiyonlama sistemi
2. Load balancing implementasyonu

### İleri Seviye
1. Mikroservis mimarisi tasarlama
2. Otomatik ölçeklendirme sistemi 