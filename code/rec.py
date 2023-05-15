import cv2
import numpy as np

# Titik-titik
titik1 = (95, 961)
titik2 = (430, 248)
titik3 = (488.09, 477.48)

# Menggambar garis antara titik pertama dan titik kedua
img = np.zeros((1000, 1000, 3), dtype=np.uint8)
cv2.line(img, titik1, titik2, (0, 255, 0), 2)

# Menggambar garis antara titik kedua dan titik ketiga
cv2.line(img, titik2, titik3, (0, 255, 0), 2)

# Menghitung sudut antara dua garis
vector1 = np.array(titik2) - np.array(titik1)
vector2 = np.array(titik3) - np.array(titik2)
angle = cv2.fastAtan2(vector2[1], vector2[0]) - cv2.fastAtan2(vector1[1], vector1[0])
if angle < 0:
    angle += 360

# Menampilkan sudut
print("Sudut antara dua garis: {:.2f} derajat".format(angle))

# Menampilkan gambar dengan garis dan sudut
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
