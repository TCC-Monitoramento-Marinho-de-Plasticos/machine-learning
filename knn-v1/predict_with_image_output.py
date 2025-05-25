import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Erro: caminho da imagem não fornecido")
        sys.exit(1)

    image_path = sys.argv[1]
    output_image_path = "/tmp/output_prediction.png"

    try:
        knn = pickle.load(open("knn_model.pkl", "rb"))
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        sys.exit(1)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro: imagem inválida")
        sys.exit(1)

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img_flattened = img.flatten()

    prediction = knn.predict([img_flattened])[0]
    label = "SEM lixo" if prediction == 0 else "COM lixo"

    plt.imshow(img, cmap='gray')
    plt.title("Predição: " + label)
    plt.axis('off')
    plt.savefig(output_image_path)
    print(output_image_path)

if __name__ == "__main__":
    main()
