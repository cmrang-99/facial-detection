import cv2
import matplotlib.pyplot as plt

imagePath = r'C:\Users\cmran\fac_rec_admission\.venv\static-test-image.jpg'

img = cv2.imread(imagePath)

img.shape
print("Image Shape:", img.shape)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_image.shape
print("Gray Image Shape:", gray_image.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# Drawing the Bounding Box
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Using Matplotlib to Display Image
plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()