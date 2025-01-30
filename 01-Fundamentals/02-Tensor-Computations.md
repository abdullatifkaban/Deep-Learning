# Tensör Hesaplamaları ve Uygulamaları

## Tensör Nedir?
Tensör, çok boyutlu bir veri yapısıdır ve lineer cebirin genişletilmiş bir kavramıdır. Skalerler, vektörler ve matrisler, tensörlerin özel durumlarıdır. Tensörler, derin öğrenmede veri temsili ve hesaplama süreçlerinde temel bir rol oynar.

- **Skaler:** Sıfır boyutlu bir tensördür (örneğin, $x = 5$).
- **Vektör:** Bir boyutlu bir tensördür (örneğin, $\mathbf{v} = [1, 2, 3]$).
- **Matris:** İki boyutlu bir tensördür (örneğin, $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$).
- **Genel Tensör:** Daha yüksek boyutlu veri yapılarıdır (örneğin, 3D bir tensör: $\mathcal{T}_{ijk}$).

## Tensör Notasyonu
Bir tensör genelde aşağıdaki şekillerde gösterilir:
- Skaler: $x$
- Vektör: $\mathbf{v}$
- Matris: $\mathbf{A}$
- 3D Tensör: $\mathcal{T}$

## Tensör İşlemleri

### 1. Tensör Toplama
Tensörlerin boyutları aynı olduğunda eleman bazında toplama yapılabilir:

$`
\mathcal{T} + \mathcal{U} = \mathcal{V}, \text{ burada } v_{ijk} = t_{ijk} + u_{ijk}
`$

### 2. Skaler ile Çarpma
Bir tensör, bir skaler ile çarpıldığında her eleman skalerle çarpılır:

$`
\mathcal{T} \cdot c = \mathcal{V}, \text{ burada } v_{ijk} = c \cdot t_{ijk}
`$

### 3. Matris Çarpımı (İki Boyutlu Tensörler için)
Matris çarpımı, satır ve sütunların iç çarpımına dayanır:

$`
\mathbf{C} = \mathbf{A} \cdot \mathbf{B}, \text{ burada } c_{ij} = \sum_k a_{ik} b_{kj}
`$

### 4. Tensör Çarpımı (Genel Tensörler için)
Daha yüksek boyutlu tensörlerde çarpım, belirli eksenlerde yapılır:

$`
(\mathcal{T} \ast \mathcal{U})_{ij} = \sum_k t_{ik} u_{kj}
`$

### 5. Tensör Dilimleme
Tensör dilimleme, tensörün alt kümelerini elde etme işlemidir. Örneğin:
- $\mathcal{T}[i, :, :]$: $i$'nci düzlemi seçer.
- $\mathcal{T}[:, j, :]$: $j$'nci sütunu seçer.

### 6. Boyut Değiştirme (Reshape)
Tensörlerin boyutları, veri kaybetmeden yeniden düzenlenebilir. Örneğin:
- $\mathcal{T} \in \mathbb{R}^{2 \times 3}$, yeniden şekillendirildiğinde $\mathcal{T} \in \mathbb{R}^{3 \times 2}$ olabilir.

## Tensörlerin Derin Öğrenmedeki Kullanımı

### 1. Veri Temsili
- **Görüntüler:** RGB formatında 3D tensörler olarak temsil edilir ($\text{Genişlik} \times \text{Yükseklik} \times \text{Kanal}$).
- **Ses Verisi:** Zamana dayalı 2D tensörler.
- **Metin Verisi:** Dizinlenmiş kelimeler veya kelime gömme (word embedding) tensörleri.

### 2. Model Parametreleri
- Ağdaki ağırlıklar ve önyargılar tensörler olarak saklanır.

### 3. Geri Yayılım (Backpropagation)
- Tensör türevleri, kayıp fonksiyonunun optimize edilmesinde kullanılır.

### 4. Tensör Kütüphaneleri
- **NumPy:** Temel tensör işlemleri için.
- **PyTorch:** Otomatik türev hesaplama ve dinamik grafikler.
- **TensorFlow:** Yüksek performanslı tensör hesaplama.

## Örnek Tensör Hesaplamaları (NumPy ile)

```python
import numpy as np

# Tensör oluşturma
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Tensör Toplama
C = A + B
print("Toplama:", C)

# Skaler Çarpma
D = 2 * A
print("Skaler Çarpma:", D)

# Matris Çarpımı
E = np.dot(A, B)
print("Matris Çarpımı:", E)
```

## Daha Fazlası İçin Kaynaklar
- [**"Deep Learning"** (Ian Goodfellow and Yoshua Bengio and Aaron Courville)](https://www.deeplearningbook.org/)
- [NumPy Resmi Belgeleri](https://numpy.org/doc/stable/)
- [PyTorch Eğitim Serisi](https://pytorch.org/tutorials/)
