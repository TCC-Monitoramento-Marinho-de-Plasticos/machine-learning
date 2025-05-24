#!/usr/bin/env python3

import sys
import pickle
import cv2
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Erro: caminho da imagem não fornecido")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Carrega modelo KNN e PCA treinados
        with open('knn_model.pkl', 'rb') as f:
            knn = pickle.load(f)
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        sys.exit(1)

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro: imagem não encontrada ou inválida")
            sys.exit(1)

        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img_flattened = img.flatten()

        img_pca = pca.transform([img_flattened])
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
        sys.exit(1)

    try:
        prediction = knn.predict(img_pca)[0]
        if prediction == 0:
            print("A imagem NÃO possui lixo")
        else:
            print("A imagem POSSUI lixo")
    except Exception as e:
        print(f"Erro ao fazer predição: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
