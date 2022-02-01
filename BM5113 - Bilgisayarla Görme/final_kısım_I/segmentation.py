from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from imutils import paths
from tqdm import tqdm
import numpy as np
import time
import cv2
import os

from PIL import Image
from unet import Unet
import sklearn.metrics as sm


def get_transfer_model(input_shape=(224, 224, 3), classes=2):

    model = Unet(input_shape=input_shape, classes=classes)
    model.summary()
    return model


def preprocess_img(img, size=(224, 224)):
    # apply opencv preprocessing
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, size)
    img = np.asarray(img, dtype=np.uint8)[:224, :224, :]
    return img


def preprocess_mask(mask, size=(224, 224)):
    # apply opencv preprocessing
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = cv2.resize(mask, size)
    mask = np.asarray(mask, dtype=np.uint8)[:224, :224]
    return mask


def load_data(
    dataset_path="./data/all",
    mode="segmentation",
    test_size=0.25,
    multi_label=False,
    num_classes=None,
):

    if not os.path.exists(dataset_path):
        raise Exception(f"[INFO] Veri klasörü yok. dataset_path:{dataset_path}")

    # Veri seti dizinindeki tüm görüntü dosyalarını yükle
    print("[INFO] veri yukleniyor...")

    classes = sorted(os.listdir(dataset_path))[:num_classes]
    class_idxs = [i for i in range(len(classes))]

    # [(imagePath, maskPath, className), ...]
    datasetPaths = [
        tuple([imagePath, maskPath, className])
        for className in classes
        for imagePath, maskPath in zip(
            sorted(
                list(paths.list_images(os.path.join(dataset_path, className, "RAW")))
            ),
            sorted(
                list(paths.list_images(os.path.join(dataset_path, className, "BINARY")))
            ),
        )
    ]
    print("num_classes:", len(classes))
    print("sample count: ", len(datasetPaths))

    images = list()
    labels = list()
    masks = list()

    ds = tqdm(datasetPaths, desc="Veri yükleniyor...")
    # Tüm RAW görüntü türlerini oku
    for imagePath, maskPath, className in ds:

        ds.set_description(
            f"imagePath: {imagePath.split(os.path.sep)[-1]}, maskPath: {maskPath.split(os.path.sep)[-1]}, className: {className}"
        )

        # Görüntüyü oku ve önişle
        # img = cv2.imread(imagePath),
        img = np.array(Image.open(imagePath).convert("RGB"))
        image = preprocess_img(img)
        image = preprocess_input(image)
        images.append(image)

        # mask = cv2.imread(maskPath)
        mask = np.array(Image.open(maskPath).convert("L"))
        mask = preprocess_mask(mask)

        if multi_label:
            multi_label_mask = np.zeros(
                (mask.shape[0], mask.shape[1], len(classes)), dtype=np.uint8
            )

            multi_label_mask[:, :, classes.index(className)] = mask

            mask = multi_label_mask
            masks.append(mask)
        else:
            masks.append(mask)

        labels.append(className)

    images = np.array(images, dtype="uint8")
    masks = np.array(masks, dtype="uint8")
    labels = np.array(labels)

    # Kategorik etiketleri dönüştür
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    # labels = to_categorical(labels)

    if mode == "segmentation":
        # veriyi %75 eğitim %25 test olmak üzere ayır.
        (trainX, testX, trainY, testY) = train_test_split(
            images, masks, test_size=test_size, stratify=labels, random_state=42
        )
    else:
        # classificaiton
        (trainX, testX, trainY, testY) = train_test_split(
            images, labels, test_size=test_size, stratify=labels, random_state=42
        )

    return trainX, testX, trainY, testY, lb


def train_model(model, trainX, trainY, testX, testY):
    # Modeli derle
    print("[INFO] model derleniyor...")
    INIT_LR = 1e-4
    EPOCHS = 20
    BS = 32
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("Derlendi !!")
    # Veriyi çeşitlendirmek (data augmentation) için aşağıdaki fonksiyon çağrılabilir
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Modelin sonradan eklenen ana kısmını eğit
    print("[INFO] Ana model egitiliyor...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS,
    )
    # Model eğitim kaybı ve başarısını çizdir
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("evaluation.png")

    return model


def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """returns confusion matrix elements i.e TN, FP, FN, TP as floats
    See example code for helper function definitions
    """
    tn, fp, fn, tp = sm.confusion_matrix(
        groundtruth_list, predicted_list, labels=[0, 1]
    ).ravel()
    tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

    return tn, fp, fn, tp


def get_prec_rec_IoU_accuracy(groundtruth_list, predicted_list):
    """returns precision, recall, IoU and accuracy metrics"""
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    IoU = tp / (tp + fp + fn)

    return prec, rec, IoU, accuracy


def get_f1_score(groundtruth_list, predicted_list):
    """Return f1 score covering edge cases"""

    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return f1_score


def get_validation_metrics(groundtruth, predicted):
    """Return all output metrics. Input is binary images"""

    u, v = np.shape(groundtruth)
    groundtruth_list = np.reshape(groundtruth, (u * v,))
    predicted_list = np.reshape(predicted, (u * v,))
    prec, rec, IoU, acc = get_prec_rec_IoU_accuracy(groundtruth_list, predicted_list)
    f1_score = get_f1_score(groundtruth_list, predicted_list)
    # print("Precision=",prec, "Recall=",rec, "IoU=",IoU, "acc=",acc, "F1=",f1_score)
    return prec, rec, IoU, acc, f1_score


def evaluate_model(model, testX, testY, lb, BS=32):

    start = time.time()
    # Test kümesi üzerindeki tahminleri belirle
    print("[INFO] model degerlendiriliyor...")
    predIdxs = model.predict(testX, batch_size=BS)
    print(time.time() - start)

    # En yüksek olasılığa denk gelen etiketin indeksini
    # tüm test görüntüleri için belirle
    predIdxs = np.argmax(predIdxs, axis=1)
    """
    # Gerçek sınıf etiketleri ve tahmin edilen etiketleri karşılaştır
    # ve düzgün bir formatta raporla
    print(
        classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_)
    )
    """


if __name__ == "__main__":
    # Veriyi yükle ve eğitim ve test olarak ayır.
    trainX, testX, trainY, testY, lb = load_data()

    # Modeli oluştur ve eğit
    # A şıkkı için tamamlanması gereken model
    cnn_model = get_transfer_model()
    # B şıkkı için oluşturulması gereken model
    # cnn_model = get_your_model()
    cnn_model = train_model(cnn_model, trainX, trainY, testX, testY)
    # Modeli değerlendir
    evaluate_model(cnn_model, testX, testY, lb)

    # Modeli diske kaydet
    print("[INFO] Mutfak araç ve gereçleri modelini diske kaydet...")
    cnn_model.save("./out/kitchen_equipments_classifier.model", save_format="h5")
    print("Bitti !")
