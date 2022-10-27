from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np 
import cv2 
import time

def preprocess_img(img):
    # apply opencv preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,  (640, 640)) 
    img = np.asarray(img, dtype=np.float)
    return img

def load_data():
    dataset = 'all'
    # Veri seti dizinindeki tüm görüntü dosyalarını yükle
    print("[INFO] veri yukleniyor...")
    # Görüntü dizinlerini oku
    imagePaths = list(paths.list_images(dataset))
    data = []
    labels = []
    # Tüm RAW görüntü türlerini oku
    for imagePath in imagePaths:
        # RAW dosya ismi ise görüntüyü okuyup data içerisine ekle
        img_type = imagePath.split('\\')[-2]
        # Dosya isminden etiketi belirle
        label = imagePath.split('\\')[-3]
        if img_type == 'RAW':
            # Görüntüyü oku ve önişle
            img = cv2.imread(imagePath)
            image = preprocess_img(img)       
            # update the data and labels lists, respectively
            data.append(image)
            labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    # veriyi %65 eğitim %35 test olmak üzere ayır.
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.35, stratify=labels, random_state=42)
   
    return trainX, testX, trainY, testY

def main():
    # Veriyi yükle ve eğitim ve test olarak ayır.
    trainX, testX, trainY, testY = load_data()

    # Kodu yaz

if __name__ == '__main__':
    main()


