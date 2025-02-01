# Nesne Tespiti

## 📍 Bölüm Haritası
- Önceki Bölüm: [Doğal Dil İşleme](02-Dogal-Dil-Isleme.md)
- Sonraki Bölüm: [Zaman Serisi](04-Zaman-Serisi.md)
- Tahmini Süre: 5-6 saat
- Zorluk Seviyesi: 🟡 Orta

## 🎯 Hedefler
- Nesne tespiti mimarilerini anlama
- Bounding box tahminlerini öğrenme
- IoU ve NMS kavramlarını kavrama
- Modern detektörleri kullanabilme

## 🎯 Öz Değerlendirme
- [ ] Temel mimarileri açıklayabiliyorum
- [ ] Bounding box hesaplayabiliyorum
- [ ] IoU ve NMS uygulayabiliyorum
- [ ] YOLO ve SSD kullanabiliyorum

## 🚀 Mini Projeler
1. Yüz Tespiti
   - OpenCV ile yüz tespiti
   - MTCNN implementasyonu
   - Performans analizi

2. Araç Tespiti
   - YOLO modeli eğitimi
   - Custom veri seti hazırlama
   - Real-time tespit

## 📑 Ön Koşullar
- CNN mimarisi bilgisi
- Python ve OpenCV deneyimi
- Temel görüntü işleme
- GPU programlama

## 🔑 Temel Kavramlar
1. Bounding Box
2. Anchor Boxes
3. IoU (Intersection over Union)
4. Non-Max Suppression

## Veri Hazırlama
🔴 İleri

💡 İpucu: Doğru veri etiketleme ve augmentasyon, model performansı için kritiktir

### 1. Veri Etiketleme
```python
def create_bounding_boxes(image, annotations):
    boxes = []
    labels = []
    
    for ann in annotations:
        x_min, y_min, width, height = ann['bbox']
        boxes.append([x_min, y_min, x_min + width, y_min + height])
        labels.append(ann['category_id'])
        
    return np.array(boxes), np.array(labels)
```

## Model Oluşturma

### 1. SSD (Single Shot Detector)
```python
# SSD modeli oluşturma
def build_ssd_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(300, 300, 3),
        include_top=False
    )
    
    # Özellik haritaları
    c3 = base_model.get_layer('block_6_expand_relu').output
    c4 = base_model.get_layer('block_13_expand_relu').output
    c5 = base_model.get_layer('Conv_1').output
    
    # Tahmin başlıkları
    outputs = []
    for feature_map in [c3, c4, c5]:
        outputs.extend([
            tf.keras.layers.Conv2D(num_classes, 3, padding='same')(feature_map),
            tf.keras.layers.Conv2D(4, 3, padding='same')(feature_map)
        ])
    
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)
```

### 2. YOLO (You Only Look Once)
```python
def build_yolo_model(num_classes):
    base_model = tf.keras.applications.DarkNet(
        include_top=False,
        weights=None,
        input_shape=(416, 416, 3)
    )
    
    x = base_model.output
    x = tf.keras.layers.Conv2D(512, 1)(x)
    x = tf.keras.layers.Conv2D(num_classes + 5, 1)(x)
    
    return tf.keras.Model(inputs=base_model.input, outputs=x)
```

## Transfer Learning

### 1. Pre-trained Model Kullanımı
```python
# TensorFlow Hub'dan model yükleme
import tensorflow_hub as hub

detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")

def detect_objects(image_path):
    image = tf.image.decode_jpeg(tf.io.read_file(image_path))
    converted_img = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
    
    result = detector(converted_img)
    
    return {
        'detection_boxes': result['detection_boxes'],
        'detection_scores': result['detection_scores'],
        'detection_classes': result['detection_classes']
    }
```

### 2. Fine-tuning
```python
# Model yükleme ve fine-tuning
model = tf.keras.applications.EfficientDet(
    weights='imagenet',
    input_shape=(512, 512, 3)
)

# Son katmanları eğitilebilir yap
for layer in model.layers[-20:]:
    layer.trainable = True

# Yeni sınıflandırma başlığı ekle
outputs = [
    tf.keras.layers.Dense(num_classes, activation='softmax')(model.output[0]),
    model.output[1]  # bounding box çıktıları
]

model = tf.keras.Model(inputs=model.input, outputs=outputs)
```

## Özel Veri Seti Eğitimi

### 1. Veri Hazırlama
```python
def prepare_dataset(image_paths, annotations):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotations))
    
    def load_and_preprocess(image_path, annotation):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [300, 300])
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        
        return image, annotation
    
    dataset = dataset.map(load_and_preprocess)
    dataset = dataset.shuffle(1000).batch(32)
    
    return dataset
```

### 2. Model Eğitimi
```python
# Loss fonksiyonları
def detection_loss(y_true, y_pred):
    # Sınıflandırma kaybı
    class_loss = tf.keras.losses.binary_crossentropy(
        y_true[..., :num_classes],
        y_pred[..., :num_classes]
    )
    
    # Bounding box kaybı
    box_loss = tf.keras.losses.huber(
        y_true[..., num_classes:],
        y_pred[..., num_classes:]
    )
    
    return class_loss + box_loss

# Model derleme ve eğitim
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=detection_loss,
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_model.h5'),
        tf.keras.callbacks.EarlyStopping(patience=5)
    ]
)
```

## Örnek Uygulamalar

### 1. Gerçek Zamanlı Nesne Tespiti
```python
def real_time_detection():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Frame'i modele uygun hale getir
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Tespit yap
        detections = detect_fn(input_tensor)
        
        # Sonuçları görselleştir
        viz_utils.visualize_boxes_and_labels_on_image_array(
            frame,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
        
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
```

### 3. Yüz Tespiti ve Tanıma
```python
def build_face_detection_model():
    # Base model (ResNet50)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Özellik piramidi (Feature Pyramid Network)
    c3 = base_model.get_layer('conv3_block4_out').output
    c4 = base_model.get_layer('conv4_block6_out').output
    c5 = base_model.get_layer('conv5_block3_out').output
    
    # FPN katmanları
    p5 = tf.keras.layers.Conv2D(256, 1)(c5)
    p4 = tf.keras.layers.Add()([
        tf.keras.layers.UpSampling2D()(p5),
        tf.keras.layers.Conv2D(256, 1)(c4)
    ])
    p3 = tf.keras.layers.Add()([
        tf.keras.layers.UpSampling2D()(p4),
        tf.keras.layers.Conv2D(256, 1)(c3)
    ])
    
    # Sınıflandırma ve lokalizasyon başlıkları
    def build_head(feature_map):
        x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(feature_map)
        cls = tf.keras.layers.Conv2D(2, 1, activation='sigmoid')(x)  # Yüz/Arka plan
        box = tf.keras.layers.Conv2D(4, 1)(x)  # Bounding box
        return cls, box
    
    outputs = []
    for feature in [p3, p4, p5]:
        cls, box = build_head(feature)
        outputs.extend([cls, box])
    
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)
```

### 4. Çoklu Nesne Takibi
```python
class MultiObjectTracker:
    def __init__(self, detector_model, max_age=1, min_hits=3, iou_threshold=0.3):
        self.detector = detector_model
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, frame):
        self.frame_count += 1
        
        # Nesne tespiti
        detections = self.detector.predict(frame)
        
        # Kalman filtresi tahminleri
        predicted_trackers = []
        for tracker in self.trackers:
            tracker.predict()
            predicted_trackers.append(tracker.get_state())
        
        # Eşleştirme matrisi
        matching_matrix = np.zeros((len(self.trackers), len(detections)))
        for i, tracker in enumerate(predicted_trackers):
            for j, detection in enumerate(detections):
                matching_matrix[i, j] = self.calculate_iou(tracker, detection)
        
        # Macar algoritması ile eşleştirme
        matches = self.hungarian_matching(matching_matrix)
        
        # Tracker güncelleme
        for tracker_idx, detection_idx in matches:
            if matching_matrix[tracker_idx, detection_idx] > self.iou_threshold:
                self.trackers[tracker_idx].update(detections[detection_idx])
            else:
                self.trackers[tracker_idx].mark_missed()
        
        # Yeni tracker'lar oluşturma
        for i in range(len(detections)):
            if i not in matches[:, 1]:
                self.create_new_tracker(detections[i])
        
        # Eski tracker'ları temizleme
        self.trackers = [t for t in self.trackers if not t.is_dead()]
        
        return self.get_tracking_results()
```

### 5. 3D Nesne Tespiti
```python
def build_3d_detection_model():
    # Point cloud girişi
    points = tf.keras.layers.Input(shape=(None, 3))
    
    # PointNet özellik çıkarıcı
    x = tf.keras.layers.Conv1D(64, 1)(points)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv1D(128, 1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv1D(1024, 1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Global features
    global_features = tf.keras.layers.MaxPooling1D()(x)
    
    # 3D bounding box tahminleri
    x = tf.keras.layers.Dense(512, activation='relu')(global_features)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Çıktılar: merkez(3), boyutlar(3), rotasyon(3), sınıf(n)
    center = tf.keras.layers.Dense(3)(x)
    dimensions = tf.keras.layers.Dense(3, activation='relu')(x)
    rotation = tf.keras.layers.Dense(3)(x)
    classification = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(
        inputs=points,
        outputs=[center, dimensions, rotation, classification]
    )
```

### 6. Semantik Bölütleme ve Nesne Tespiti
```python
def build_mask_rcnn():
    # Backbone
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(None, None, 3)
    )
    
    # FPN
    c2 = backbone.get_layer('conv2_block3_out').output
    c3 = backbone.get_layer('conv3_block4_out').output
    c4 = backbone.get_layer('conv4_block6_out').output
    c5 = backbone.get_layer('conv5_block3_out').output
    
    # RPN (Region Proposal Network)
    def build_rpn(feature_map, anchor_scales):
        shared = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(feature_map)
        
        # Sınıflandırma (nesne/arka plan)
        cls = tf.keras.layers.Conv2D(len(anchor_scales) * 2, 1, activation='sigmoid')(shared)
        
        # Bounding box regresyonu
        reg = tf.keras.layers.Conv2D(len(anchor_scales) * 4, 1)(shared)
        
        return cls, reg
    
    # Mask branch
    def build_mask_head(feature_map):
        x = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(feature_map)
        x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
        return x
    
    # Model çıktıları
    rpn_outputs = []
    mask_outputs = []
    
    for feature_map in [c2, c3, c4, c5]:
        cls, reg = build_rpn(feature_map, [8, 16, 32])
        rpn_outputs.extend([cls, reg])
        mask_outputs.append(build_mask_head(feature_map))
    
    return tf.keras.Model(
        inputs=backbone.input,
        outputs=rpn_outputs + mask_outputs
    )
```

## 📚 Önerilen Kaynaklar
- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [Faster R-CNN Tutorial](https://www.tensorflow.org/tutorials/images/object_detection)

## ✍️ Alıştırmalar
### Başlangıç Seviyesi
1. SSD modeli ile basit nesne tespiti
2. Webcam ile gerçek zamanlı tespit

### Orta Seviye
1. YOLO modelini özel veri setinde eğitme
2. Model optimizasyonu ve FPS artırma

### İleri Seviye
1. Instance segmentation modeli geliştirme
2. 3D nesne tespiti sistemi oluşturma 