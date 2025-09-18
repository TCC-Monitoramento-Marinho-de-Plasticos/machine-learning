import matplotlib.pyplot as plt
from skimage import color, io, exposure
from skimage.feature import hog

# Carregar imagem (substitua pelo caminho da sua)
image = color.rgb2gray(io.imread('plastic_tp.jpg'))

# Extrair HOG
features, hog_image = hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm='L2-Hys'
)

# Reescalar o hog_image para aumentar o contraste
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Exibir
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Imagem Original')
ax1.axis('off')

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG')
ax2.axis('off')

plt.show()
