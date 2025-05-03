# Bölüm 8: Çekişmeli Üretici Ağlar (Generative Adversarial Networks - GANs)

## 8.1 GAN'lara Giriş

Çekişmeli Üretici Ağlar (GAN'lar), Ian Goodfellow ve arkadaşları tarafından 2014 yılında tanıtılan, denetimsiz öğrenme (unsupervised learning) sınıfına giren güçlü bir derin öğrenme modelidir. GAN'ların temel amacı, mevcut veri setine benzer yeni veriler üretmektir.

GAN'lar temel olarak iki yapay sinir ağından oluşur:

1.  **Üretici (Generator):** Rastgele gürültüden (random noise) başlayarak sahte veri örnekleri üretmeye çalışır. Amacı, ürettiği sahte verilerin gerçek verilere o kadar benzemesini sağlamaktır ki, Ayırt Edici bu sahte verileri gerçeklerinden ayıramaz hale gelsin.
2.  **Ayırt Edici (Discriminator):** Hem gerçek veri setinden gelen örnekleri hem de Üretici tarafından üretilen sahte örnekleri alır. Görevi, kendisine verilen bir örneğin gerçek mi yoksa sahte mi olduğunu ayırt etmektir.

Bu iki ağ, birbirleriyle sürekli bir "oyun" veya "çekişme" halindedir:

*   Üretici, Ayırt Edici'yi kandıracak kadar gerçekçi sahte veriler üretmeye çalışır.
*   Ayırt Edici, gerçek ve sahte verileri ayırt etme konusunda giderek daha iyi olmaya çalışır.

Bu çekişmeli süreç sonunda, Üretici oldukça gerçekçi veriler üretebilen bir model haline gelir.

**Kullanım Alanları:**

*   Görüntü Üretme (Image Generation)
*   Görüntüden Görüntüye Çeviri (Image-to-Image Translation)
*   Metin Üretme (Text Generation)
*   Veri Artırma (Data Augmentation)
*   Süper Çözünürlük (Super-Resolution)

Sonraki bölümlerde GAN'ların çalışma mekanizmasını daha detaylı inceleyecek ve pratik uygulamalar yapacağız.
