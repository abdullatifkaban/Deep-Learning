# Model Monitoring ve MLOps

## 📍 Bölüm Haritası
- Önceki Bölüm: [Model Deployment](01-Model-Deployment.md)
- Tahmini Süre: 5-6 saat
- Zorluk Seviyesi: 🟡 Orta

## 🎯 Hedefler
- Model performansını izleme
- Veri ve model driftini tespit etme
- Metrik toplama ve analiz
- Alerting sistemleri kurma

## 🎯 Öz Değerlendirme
- [ ] Monitoring sistemleri kurabiliyorum
- [ ] Drift tespiti yapabiliyorum
- [ ] Metrikleri analiz edebiliyorum
- [ ] Alerting sistemleri geliştirebiliyorum

## 🚀 Mini Projeler
1. Metrik Dashboard
   - Prometheus kurulumu
   - Grafana dashboard
   - Metrik toplama

2. Drift Detection
   - Veri drift analizi
   - Model drift tespiti
   - Otomatik retraining

## 📑 Ön Koşullar
- Model deployment bilgisi
- Python ve monitoring araçları
- İstatistik temelleri
- Linux sistem yönetimi

## 🔑 Temel Kavramlar
1. Model Drift
2. Metric Collection
3. Performance Monitoring
4. Alerting Systems

## Model İzleme
> Zorluk Seviyesi: 🔴 İleri

> 💡 İpucu: Monitoring sistemini baştan düzgün kurmak, sonradan yaşanacak sorunları önler

### 1. Prometheus ile Metrik Toplama
```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrik tanımlamaları
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Prediction latency')
```

## Model Performans İzleme

### 1. Metrik Toplama
```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ModelMonitor:
    def __init__(self, model, threshold=0.8):
        self.model = model
        self.threshold = threshold
        self.metrics_history = []
        
    def collect_metrics(self, X, y_true):
        y_pred = self.model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred > 0.5),
            'precision': precision_recall_fscore_support(y_true, y_pred > 0.5)[0].mean(),
            'recall': precision_recall_fscore_support(y_true, y_pred > 0.5)[1].mean(),
            'timestamp': pd.Timestamp.now()
        }
        self.metrics_history.append(metrics)
        return metrics
```

### 2. Prometheus ve Grafana Entegrasyonu
```python
from prometheus_client import start_http_server, Gauge

class MetricsExporter:
    def __init__(self, port=8000):
        self.accuracy_gauge = Gauge('model_accuracy', 'Model Accuracy')
        self.latency_gauge = Gauge('model_latency', 'Prediction Latency')
        start_http_server(port)
    
    def export_metrics(self, metrics):
        self.accuracy_gauge.set(metrics['accuracy'])
        self.latency_gauge.set(metrics['latency'])
```

## Model Versiyonlama

### 1. Git ile Model Versiyonlama
```python
import git
from datetime import datetime

class ModelVersionControl:
    def __init__(self, repo_path):
        self.repo = git.Repo(repo_path)
        
    def save_model_version(self, model, metrics):
        # Model kaydet
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/model_v{version}.h5"
        model.save(model_path)
        
        # Git'e ekle ve commit
        self.repo.index.add([model_path])
        self.repo.index.commit(f"Model version {version} - Accuracy: {metrics['accuracy']:.4f}")
        
        # Tag oluştur
        self.repo.create_tag(f"v{version}")
```

### 2. DVC (Data Version Control)
```python
import dvc.api

def version_data_and_model():
    # Veri versiyonlama
    with dvc.api.open(
        'data/training.csv',
        mode='rb',
        rev='v1.0'
    ) as f:
        training_data = pd.read_csv(f)
    
    # Model ve veri bağlantısı
    dvc.api.make_checkpoint(
        'Model trained with data v1.0',
        metrics={'accuracy': 0.95}
    )
```

## A/B Testing

### 1. Test Konfigürasyonu
```python
class ABTestingManager:
    def __init__(self, models, traffic_split):
        self.models = models
        self.traffic_split = traffic_split
        self.results = {model_id: [] for model_id in models.keys()}
        
    def route_request(self, request_id):
        # Request yönlendirme
        split_point = request_id % 100
        cumulative = 0
        for model_id, split in self.traffic_split.items():
            cumulative += split * 100
            if split_point < cumulative:
                return model_id
```

### 2. Sonuç Analizi
```python
def analyze_ab_test_results(results_a, results_b):
    from scipy import stats
    
    # İstatistiksel analiz
    t_stat, p_value = stats.ttest_ind(results_a, results_b)
    
    # Effect size hesaplama
    cohens_d = (np.mean(results_a) - np.mean(results_b)) / np.sqrt(
        (np.var(results_a) + np.var(results_b)) / 2
    )
    
    return {
        'p_value': p_value,
        'effect_size': cohens_d,
        'significant': p_value < 0.05
    }
```

## Otomatik Yeniden Eğitim

### 1. Veri Kalitesi Kontrolü
```python
class DataQualityChecker:
    def __init__(self, rules):
        self.rules = rules
        
    def check_data_quality(self, data):
        results = []
        for rule in self.rules:
            result = rule(data)
            results.append(result)
        return all(results)
    
    @staticmethod
    def missing_value_rule(data, threshold=0.1):
        missing_ratio = data.isnull().sum().mean() / len(data)
        return missing_ratio < threshold
```

### 2. Otomatik Eğitim Pipeline
```python
class AutomaticTrainingPipeline:
    def __init__(self, model_builder, data_collector, quality_checker):
        self.model_builder = model_builder
        self.data_collector = data_collector
        self.quality_checker = quality_checker
        
    def run_pipeline(self):
        # Yeni veri topla
        new_data = self.data_collector.collect()
        
        # Veri kalitesi kontrolü
        if not self.quality_checker.check_data_quality(new_data):
            raise ValueError("Data quality checks failed")
        
        # Model eğitimi
        model = self.model_builder()
        model.fit(new_data)
        
        # Performans değerlendirme
        metrics = self.evaluate_model(model, new_data)
        
        if self.should_deploy(metrics):
            self.deploy_model(model)
```

## CI/CD Pipeline

### 1. GitHub Actions
```yaml
name: Model CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
    - name: Train and evaluate model
      run: |
        python train.py
        python evaluate.py
```

### 2. Model Deployment Pipeline
```python
class DeploymentPipeline:
    def __init__(self, stages):
        self.stages = stages
    
    def run(self, model, config):
        for stage in self.stages:
            try:
                stage.execute(model, config)
            except Exception as e:
                self.rollback()
                raise e
    
    def rollback(self):
        for stage in reversed(self.stages):
            stage.rollback()
```

## Örnek Uygulamalar

### 3. Özel Metrik Toplayıcı
```python
class CustomMetricCollector:
    def __init__(self):
        self.prediction_latency = Summary('model_prediction_latency_seconds', 
                                        'Time spent processing prediction')
        self.prediction_counter = Counter('model_predictions_total',
                                        'Number of predictions', 
                                        ['model', 'status'])
        self.feature_histogram = Histogram('model_feature_values',
                                         'Distribution of feature values',
                                         ['feature_name'])
        
    def record_prediction(self, model_name, duration, status='success'):
        self.prediction_latency.observe(duration)
        self.prediction_counter.labels(model=model_name, status=status).inc()
    
    def record_features(self, features_dict):
        for feature_name, value in features_dict.items():
            self.feature_histogram.labels(feature_name=feature_name).observe(value)
```

### 4. Veri Kalitesi Monitörü
```python
class DataQualityMonitor:
    def __init__(self, schema):
        self.schema = schema
        self.violations = defaultdict(list)
        
    def validate_data(self, data):
        validation_results = {
            'missing_values': self.check_missing_values(data),
            'type_violations': self.check_data_types(data),
            'range_violations': self.check_value_ranges(data),
            'unique_violations': self.check_unique_constraints(data)
        }
        
        # Prometheus metriklerini güncelle
        self.update_metrics(validation_results)
        
        return validation_results
    
    def check_missing_values(self, data):
        missing = data.isnull().sum()
        return {col: count for col, count in missing.items() if count > 0}
    
    def check_data_types(self, data):
        violations = {}
        for col, expected_type in self.schema['types'].items():
            if not all(isinstance(x, expected_type) for x in data[col].dropna()):
                violations[col] = f"Expected {expected_type.__name__}"
        return violations
```

### 5. Model Performans Analizi
```python
class ModelPerformanceAnalyzer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics_history = defaultdict(list)
        self.alerts = []
        
    def update_metrics(self, y_true, y_pred, timestamp=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Metrikleri kaydet
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append({
                'value': value,
                'timestamp': timestamp or datetime.now()
            })
        
        # Performans kontrolü
        self.check_performance_degradation()
        
        return metrics
    
    def check_performance_degradation(self):
        for metric_name, history in self.metrics_history.items():
            recent_values = [h['value'] for h in history[-10:]]
            if len(recent_values) >= 10:
                trend = self.calculate_trend(recent_values)
                if trend < -0.1:  # Negatif trend
                    self.alerts.append({
                        'metric': metric_name,
                        'message': f"Performance degradation detected in {metric_name}",
                        'timestamp': datetime.now()
                    })
```

### 6. Anomali Detektörü
```python
class AnomalyDetector:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.baseline = None
        self.threshold = None
        
    def fit(self, historical_data):
        # İstatistiksel özellikleri hesapla
        self.baseline = {
            'mean': np.mean(historical_data, axis=0),
            'std': np.std(historical_data, axis=0),
            'q1': np.percentile(historical_data, 25, axis=0),
            'q3': np.percentile(historical_data, 75, axis=0)
        }
        
        # IQR tabanlı threshold
        iqr = self.baseline['q3'] - self.baseline['q1']
        self.threshold = {
            'lower': self.baseline['q1'] - 1.5 * iqr,
            'upper': self.baseline['q3'] + 1.5 * iqr
        }
    
    def detect(self, data):
        anomalies = {
            'point_anomalies': self.detect_point_anomalies(data),
            'pattern_anomalies': self.detect_pattern_anomalies(data),
            'trend_anomalies': self.detect_trend_anomalies(data)
        }
        
        return anomalies
    
    def detect_point_anomalies(self, data):
        return np.logical_or(
            data < self.threshold['lower'],
            data > self.threshold['upper']
        )
```

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. Temel metrik toplama sistemi
2. Model performans dashboard'u

### Orta Seviye
1. Drift detection sistemi
2. Otomatik model güncelleme

### İleri Seviye
1. Tam MLOps pipeline kurulumu
2. Multi-model monitoring sistemi 