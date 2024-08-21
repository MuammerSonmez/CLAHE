import cv2
import numpy as np
import matplotlib.pyplot as plt

# İstediğiniz görüntünün Dizinine göre yerini belirtip görüntünüzü seçiniz (grayscale)
image = cv2.imread('images/sourceImages/brainImage.png', cv2.IMREAD_GRAYSCALE)

# CLAHE objesi oluşturuyoruz bu bizim için otomatik histogram eşitlemesi yapıyor
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)
#ClipLimit = Eşik değerin tanımlanması için kullanılır
#tlieGridSize =  Satır ve sütundaki döşeme sayısını tanımlamamız için kullanılır

# Orijinal görüntü ve CLAHE sonuçlarını göster
plt.figure(figsize=(10, 5))

# Orijinal görüntü
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Orijinal Görüntü')
plt.axis('off')

# CLAHE uygulanmış görüntü
plt.subplot(1, 2, 2)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Uygulanmış Görüntü')
plt.axis('off')

plt.show()

#İki görüntünün de histogramlarını karşılaştır
plt.figure(figsize=(10, 5))

# Orijinal görüntünün histogramı
plt.subplot(1, 2, 1)
plt.hist(image.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.6)
plt.title('Orijinal Görüntü Histogramı')

# CLAHE uygulanmış görüntünün histogramı
plt.subplot(1, 2, 2)
plt.hist(clahe_image.ravel(), bins=256, range=[0, 256], color='green', alpha=0.6)
plt.title('CLAHE Uygulanmış Görüntü Histogramı')

#Burada dosyayı kaydetmek istediğimiz yeri seçiyoruz
cv2.imwrite("images/downloadImages/claheBrain.png", clahe_image)

plt.show()

""" NOT:Programı kullanmadan önce fotoğrafın asıl histogramını görüp ona göre 
    tileGridSize ve clipLimit parametrelerini atamanızı tavsiye ederim. 
    Fotoğrafların Threshold değerleri farklı olması gerekebilir. 

    NOT2:tileGridSize'ı mümkün olduğunca parametre olarak 8, 8 kullanmanızı tavsiye ederim.
    Bazı fotoğraflar için bu değerler arttırılıp azaltılabilir.

                                                                                                """