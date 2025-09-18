import pickle
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Carregue o modelo do arquivo
filename = 'knn_model.pkl'
knn = pickle.load(open(filename, 'rb'))


img_teste = cv2.imread('plastic (1044).jpg', cv2.IMREAD_GRAYSCALE)
img_teste = cv2.resize(img_teste, (128,128))
img_teste = img_teste / 255.0
gray_centered = img_teste - np.mean(img_teste, axis=0)
pca = PCA(n_components=50)
transformed = pca.fit_transform(gray_centered)
reconstructed = pca.inverse_transform(transformed)
reconstructed += np.mean(img_teste, axis=0)
img_flattened = reconstructed.flatten() 

print("A imagem não possui lixo" if knn.predict([img_flattened]) == 0 else "A imagem possui Lixo")
plt.imshow(img_teste, cmap='gray')