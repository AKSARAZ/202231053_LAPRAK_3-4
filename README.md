# 202231053_LAPRAK_3-4

## Penjelasan Program


```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import data
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix, graycoprops
```

Pertama Import Library yang diperlukan

```python
img = cv2.imread('1.jpg')

cv2.imshow("gambar parkir", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Menggunakan OpenCV (cv2) untuk memuat dan menampilkan gambar '1.jpg'.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 100, 150)

cv2.imshow("gambar parkir", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Mengubah gambar ke citra grayscale dan melakukan deteksi tepi menggunakan metode Canny.

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
ax = axs.ravel()

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("gambar asli")

ax[1].imshow(edges, cmap="gray")
ax[1].set_title("gambar yang udah")
```

Menggunakan matplotlib untuk membuat subplot dan menampilkan gambar asli dalam grayscale serta gambar yang telah diproses dengan deteksi tepi.

```python
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
image_line = img.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_line, (x1, y1), (x2, y2), (0, 255, 0), 2)

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
ax = axs.ravel()

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("gambar asli")

ax[1].imshow(image_line, cmap="gray")
ax[1].set_title("gambar yang udah")
```

Menggunakan Transformasi Hough untuk mendeteksi garis-garis pada gambar tepi dan menambahkan garis yang terdeteksi ke gambar asli.

```python
image = skimage.data.coffee()
img_hsv = rgb2hsv(image)

mean = np.mean(img_hsv.ravel())
std = np.std(img_hsv.ravel())

h_channel = (img_hsv[:, :, 0] * 255).astype(np.uint8)
s_channel = (img_hsv[:, :, 1] * 255).astype(np.uint8)
v_channel = (img_hsv[:, :, 2] * 255).astype(np.uint8)

glcm_h = graycomatrix(h_channel, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
glcm_s = graycomatrix(s_channel, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
glcm_v = graycomatrix(v_channel, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

contrast_h = graycoprops(glcm_h, 'contrast')[0, 0]
dissimilarity_h = graycoprops(glcm_h, 'dissimilarity')[0, 0]
homogeneity_h = graycoprops(glcm_h, 'homogeneity')[0, 0]
energy_h = graycoprops(glcm_h, 'energy')[0, 0]
correlation_h = graycoprops(glcm_h, 'correlation')[0, 0]

contrast_s = graycoprops(glcm_s, 'contrast')[0, 0]
dissimilarity_s = graycoprops(glcm_s, 'dissimilarity')[0, 0]
homogeneity_s = graycoprops(glcm_s, 'homogeneity')[0, 0]
energy_s = graycoprops(glcm_s, 'energy')[0, 0]
correlation_s = graycoprops(glcm_s, 'correlation')[0, 0]

contrast_v = graycoprops(glcm_v, 'contrast')[0, 0]
dissimilarity_v = graycoprops(glcm_v, 'dissimilarity')[0, 0]
homogeneity_v = graycoprops(glcm_v, 'homogeneity')[0, 0]
energy_v = graycoprops(glcm_v, 'energy')[0, 0]
correlation_v = graycoprops(glcm_v, 'correlation')[0, 0]

print(f'Contrast H Channel: {contrast_h}')
print(f'Dissimilarity H Channel: {dissimilarity_h}')
print(f'Homogeneity H Channel: {homogeneity_h}')
print(f'Energy H Channel: {energy_h}')
print(f'Correlation H Channel: {correlation_h}')

print(f'Contrast S Channel: {contrast_s}')
print(f'Dissimilarity S Channel: {dissimilarity_s}')
print(f'Homogeneity S Channel: {homogeneity_s}')
print(f'Energy S Channel: {energy_s}')
print(f'Correlation S Channel: {correlation_s}')

print(f'Contrast V Channel: {contrast_v}')
print(f'Dissimilarity V Channel: {dissimilarity_v}')
print(f'Homogeneity V Channel: {homogeneity_v}')
print(f'Energy V Channel: {energy_v}')
print(f'Correlation V Channel: {correlation_v}')
```

Menggunakan skimage untuk memuat gambar kopi ('coffee') dan mengonversinya ke ruang warna HSV. Kemudian, ekstraksi nilai rata-rata dan standar deviasi dari seluruh nilai dalam gambar HSV. Dilanjutkan dengan memisahkan saluran H, S, dan V dari gambar HSV dan menghitung GLCM untuk setiap saluran dengan jarak 1 pixel dan sudut 0 derajat. Setelah itu, ekstraksi berbagai properti GLCM seperti kontras, disimilaritas, homogenitas, energi, dan korelasi untuk masing-masing saluran H, S, dan V dilakukan untuk menganalisis tekstur gambar.
