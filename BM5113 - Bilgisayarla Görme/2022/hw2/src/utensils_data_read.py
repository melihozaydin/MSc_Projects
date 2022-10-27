from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import cv2
import time
from tqdm import tqdm
import os


def preprocess_img(img):
    # apply opencv preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = np.asarray(img, dtype=np.float64)
    return img


def load_data(dataset_path="all", num_classes=None):
    class_names = os.listdir(dataset_path)

    dataset = {"image_paths": [], "labels": [], "class_names": []}
    for class_name in class_names[:num_classes]:
        img_root_dir = os.path.join(dataset_path, class_name, "RAW")

        for imagePath in sorted(os.listdir(img_root_dir)):
            # Dosya isminden etiketi belirle
            label = np.array(class_names.index(class_name))

            dataset["image_paths"].append(os.path.join(img_root_dir, imagePath))
            dataset["labels"].append(label)
            dataset["class_names"].append(class_name)

    return dataset


def split_dataset(dataset, test_size=0.25, random_state=42):
    # veriyi %65 eğitim %35 test olmak üzere ayır.
    (trainX, testX, trainY, testY) = train_test_split(
        dataset["image_paths"],
        dataset["labels"],
        test_size=test_size,
        stratify=dataset["labels"],
        random_state=random_state,
    )

    return trainX, testX, trainY, testY


def main():
    # Veriyi yükle ve eğitim ve test olarak ayır.
    dataset = load_data(dataset_path="BM5113 - Bilgisayarla Görme/hw2/data/all", num_classes=None)
    trainX, testX, trainY, testY = split_dataset(dataset, test_size=0.25, random_state=42)

    print("[INFO] loading images...")
    # Verileri yükle
    data = []
    labels = []
    for imagePath in tqdm(trainX):
        image = cv2.imread(imagePath)
        image = preprocess_img(image)
        data.append(image)
        labels.append(trainY[trainX.index(imagePath)])
    
    return data, labels

if __name__ == "__main__":
    main()
