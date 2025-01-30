# Lineer Cebirin Temel Kavramları

## Lineer Cebir Nedir?
Lineer cebir, vektörler ve matrisler gibi lineer (doğrusal) matematiksel yapıları inceleyen bir matematik dalıdır. Derin öğrenme modellerinde verilerin temsil edilmesi ve işlenmesi için temel bir araçtır. Sinir ağlarının temellerini anlamak için lineer cebirin temel kavramlarını bilmek gereklidir.

## Temel Kavramlar

### 1. Skaler, Vektör, Matris ve Tensör
- **Skaler:** Tek bir sayı (örneğin, $x = 5$).
- **Vektör:** Bir sayı dizisi, bir boyutlu dizi (örneğin, $\mathbf{v} = [1, 2, 3]$).
- **Matris:** Bir sayı tablosu, iki boyutlu dizi (örneğin, $`\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`$).
- **Tensör:** Daha yüksek boyutlu diziler.

### 2. Vektörler Üzerinde İşlemler
- **Toplama:** İki vektör, eleman bazında toplanır:
  $ \mathbf{u} + \mathbf{v} = [u_1 + v_1, u_2 + v_2, \dots] $
- **Çıkarma:** İki vektör eleman bazında çıkarılır:
  $$ \mathbf{u} - \mathbf{v} = [u_1 - v_1, u_2 - v_2, \dots] $$
- **Skaler Çarpım:** Bir vektör, bir skalerle çarpılır:
  $$ c \cdot \mathbf{v} = [c \cdot v_1, c \cdot v_2, \dots] $$

### 3. Matrisler
- **Matris Toplama:**
  $$ \mathbf{A} + \mathbf{B} = \begin{bmatrix} a_{11} + b_{11} & a_{12} + b_{12} \\ a_{21} + b_{21} & a_{22} + b_{22} \end{bmatrix} $$
- **Skaler Çarpma:**
  $$ c \cdot \mathbf{A} = \begin{bmatrix} c \cdot a_{11} & c \cdot a_{12} \\ c \cdot a_{21} & c \cdot a_{22} \end{bmatrix} $$
- **Matris Çarpımı:**
  $$ \mathbf{C} = \mathbf{A} \cdot \mathbf{B}, \text{ burada } c_{ij} = \sum_k a_{ik} b_{kj} $$

### 4. Doğrusal Bağımlılık ve Bağımsızlık
- Bir grup vektör, doğrusal olarak bağımsızdır eğer:
  $$ c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_n \mathbf{v}_n = \mathbf{0} $$
  eşitliği sadece $c_1 = c_2 = \dots = c_n = 0$ olduğunda sağlanır.

### 5. Matris Determinantı
- Kare matrisler için tanımlıdır ve şu şekilde hesaplanır:
  $$ \det(\mathbf{A}) = a_{11}a_{22} - a_{12}a_{21} \text{ (2x2 matrisler için)} $$

### 6. Ters Matris
- Bir kare matrisin tersi $\mathbf{A}^{-1}$, şu koşulu sağlar:
  $$ \mathbf{A} \cdot \mathbf{A}^{-1} = \mathbf{I} $$
- Tersi bulmak için:
  $$ \mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix} \text{ (2x2 matrisler için)} $$

### 7. Özvektörler ve Özdeğerler
- $\mathbf{A}$ bir kare matris olsun. $\mathbf{v}$ bir özvektör ve $\lambda$ bir özdeğer olacak şekilde şu eşitlik sağlanır:
  $$ \mathbf{A} \mathbf{v} = \lambda \mathbf{v} $$

## Lineer Cebirin Derin Öğrenmedeki Önemi
1. **Veri Temsili:** Görüntüler, metinler ve ses gibi veriler tensörlerle temsil edilir.
2. **Optimizasyon:** Matris çarpımları ve türevler, model eğitiminin temelidir.
3. **Ağ Yapısı:** Sinir ağlarının ağırlıkları ve bağlantıları matrislerle ifade edilir.

## Daha Fazlası İçin Kaynaklar
- [**"Linear Algebra and Its Applications"** (Gilbert Strang)](https://rksmvv.ac.in/wp-content/uploads/2021/04/Gilbert_Strang_Linear_Algebra_and_Its_Applicatio_230928_225121.pdf)
- [3Blue1Brown Lineer Cebir Serisi](https://www.3blue1brown.com/)
- [NumPy](https://numpy.org/doc/stable/) ve [TensorFlow](https://www.tensorflow.org/learn?hl=tr) dökümantasyonları
