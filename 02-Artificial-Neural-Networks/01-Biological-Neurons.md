# Sinir Hücreleri ve Biyolojik İlham

## 1. Sinir Hücrelerinin Yapısı

Sinir hücreleri (nöronlar), biyolojik sinir sisteminin temel bileşenleridir. Beynimizde yaklaşık 86 milyar nöron bulunur ve bu nöronlar birbirleriyle karmaşık bağlantılar kurarak bilgi işlemlerini gerçekleştirirler.

Bir sinir hücresi temel olarak üç ana bileşenden oluşur:

- **Dendritler:** Diğer nöronlardan gelen sinyalleri alır.
- **Hücre Gövdesi (Soma):** Gelen sinyalleri işler ve gerekli hesaplamaları yapar.
- **Akson:** İşlenen sinyali diğer nöronlara ileten uzun bir çıkıntıdır.

Bu yapı sayesinde nöronlar, elektriksel ve kimyasal sinyaller aracılığıyla birbirleriyle iletişim kurarlar.

![Biyilogical neurons](https://miro.medium.com/v2/resize:fit:1400/1*K1ee1SzB0lxjIIo7CGI7LQ.png) 

[Kaynak](https://miro.medium.com/v2/resize:fit:1400/1*K1ee1SzB0lxjIIo7CGI7LQ.png)

## 2. Nöronların Çalışma Mekanizması

Nöronların çalışma prensibi, **aksiyon potansiyeli** adı verilen bir elektriksel uyarıya dayanır:

1. **Uyarının Alınması:** Dendritler, diğer nöronlardan gelen kimyasal sinyalleri alır.
2. **İşleme ve Karar:** Soma, aldığı sinyallerin toplamını hesaplar. Eğer belirli bir eşik değer aşılırsa, nöron aktive olur.
3. **İletim:** Akson, aksiyon potansiyeli üreterek elektriksel sinyali diğer nöronlara iletir.

Bu süreç, bilgi işlemenin temel mekanizmasını oluşturur ve yapay sinir ağlarının geliştirilmesine ilham kaynağı olmuştur.

## 3. Biyolojik Nöronlardan Yapay Nöronlara

Biyolojik nöronların işleyişi, yapay sinir ağlarının temel modelini oluşturmuştur. Yapay nöronlar, aşağıdaki bileşenlerden oluşur:


- **Girişler \( x_1, x_2, ..., x_n \)**: Çevreden alınan veriler.
- **Ağırlıklar \( w_1, w_2, ..., w_n \)**: Her girişin önemini belirleyen katsayılar.
- **Net Girdi (Toplam Fonksiyon):**

  $`
  z = \sum_{i=1}^{n} w_i x_i + b
  `$
- **Aktivasyon Fonksiyonu:** Çıktıyı sınırlayan ve modele doğrusal olmayan özellikler kazandıran matematiksel fonksiyon. Örnek:

  $`
  y = f(z)
  `$

  Burada $f(z)$ yaygın olarak kullanılan sigmoid, ReLU veya tanh fonksiyonları olabilir.

![Artificial Neuron](https://upload.wikimedia.org/wikipedia/commons/c/c6/Artificial_neuron_structure.svg)

[Kaynak](https://upload.wikimedia.org/wikipedia/commons/c/c6/Artificial_neuron_structure.svg)

## 4. Sonuç

Biyolojik nöronların çalışma prensiplerinden ilham alınarak geliştirilen yapay nöronlar, günümüzde makine öğrenimi ve derin öğrenme modellerinin temelini oluşturmaktadır. Sinir ağlarının katmanlar halinde düzenlenmesi, karmaşık problemleri çözebilme yeteneğini artırmaktadır.

## Daha Fazlası İçin Kaynaklar
- [**"Deep Learning"** (Ian Goodfellow and Yoshua Bengio and Aaron Courville)](https://www.deeplearningbook.org/)
- [**"Pattern Recognition and Machine Learning"** (Christopher Bishop)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)