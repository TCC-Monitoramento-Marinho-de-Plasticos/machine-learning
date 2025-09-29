import cv2
import pickle
import numpy as np
from skimage.feature import hog

# carregar modelo salvo com pickle
with open("svm_model.pkl", "rb") as f:
    clf = pickle.load(f)

def predict_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erro ao abrir a imagem: {image_path}")

    img = cv2.resize(img, (256, 256)).astype("float32") / 255.0

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    X = np.array([features])
    pred = clf.predict(X)[0]

    label = "plastic" if pred == 1 else "no-plastic"
    return label



# exemplo de uso
image_path = "clean5.png"
label = predict_single_image(image_path)

print(f"Classe prevista: {label}")

