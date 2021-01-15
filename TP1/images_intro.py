# -*- coding: utf-8 -*-
# Attentions : faire adjustments necessaires pour les differentes versions de Python


# petite fonction pour rafraichire une figure et faire un pause jusqu'a ce que la touche entrée soit pressée
import skimage.io  # import de la librairie scikit-image
import skimage.filters
from PIL import ImageOps, Image, ImageFilter
import math
from skimage import transform as tf
import scipy.ndimage  # http://docs.scipy.org/doc/scipy/reference/ndimage.html
import matplotlib.pyplot as plt  # import de methodes d'affichage
import numpy
import PIL.Image  # import de la librairie Pillow ou PIL #documentation ici http://effbot.org/imagingbook/pil-index.htm and Pillow Fork here https://pillow.readthedocs.org/en/latest/
import cv2  # import de la librairie opencv, documentation ici http://docs.opencv.org/trunk/doc/py_tutorials/py_tutorials.html
import skimage.exposure
from skimage import feature


def pause():
    plt.draw()
    plt.pause(0.001)
    input("Press Enter to continue...")

# --------------- Import des librairies qui seront utilisées ----------


# ---------  ouverture d'une image -----------------
print("---------  ouverture d'une image ----------------")
# avec scikit-image:
print("Avec scikit-image ")
im = skimage.io.imread('einstein.jpg')
print(im.shape)  # taille de l'image
print(im.dtype)  # type des elements de l'image

# avec opencv
print("Avec opencv")
im = cv2.imread('einstein.jpg')
# notez que l'on a un tableau 3D malgrès le fait que l'image est en niveau de gris
print(im.shape)
print(im.dtype)  # type des elements de l'image
im = cv2.imread('einstein.jpg', 0)  # on recupère le premier en niveaux de gris
print(im.shape)

# avec PIL
print("Avec PIL")
imPIL = PIL.Image.open('einstein.jpg')
# PIL n'utilise pas pas default de ableau Numpy, mais sa propore classe d'image
print(imPIL)
# cela a l'avantage de permettre d'acceder à la liste des méthodes specifiques
# aux image par autocompletion en tapant "im."
print(imPIL.size)  # taille de l'image
print(imPIL.bits)  # nombre de bits par elements de l'image
# conversion d'une image PIL vers Numpy
im2 = numpy.array(imPIL)
# conversion d'un tableau numpy vers une image PIL
arr = cv2.imread('einstein.jpg')
imPIL = PIL.Image.fromarray(arr)

# -------- affichage d'une image -------------------

# affichage bloquant
im = skimage.io.imread('einstein.jpg')
fig = plt.figure()
plt.imshow(im)
plt.title('Fermez la figure pour continue')
plt.show()  # ouvre la fenêtre et affiche l'image , stope l'execution jusqu'a ce que l'on ferme la fenetre

# affichage non bloquant
plt.ion()  # passe en mode interactif: l'execution continue après l'affichage
fig = plt.figure()
# cette fois l'execution continue, mais on perd la possibilié de zoomer etc
plt.imshow(im)
plt.title('Appuiez sur Entree pour continuer')
pause()
# pauseFigure(fig)


# affichage sous partimenté avec affichage en niveau de gris à droite
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.imshow(im, cmap=plt.cm.Greys_r)
plt.title('Example de subplot ')
pause()

# affichage sans la grille
ax = plt.subplot(1, 1, 1)
plt.imshow(im, cmap=plt.cm.Greys_r)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title(u"Example d'axes cachés ")
pause()


# ajout de la barre des couleurs
ax = plt.subplot(1, 1, 1)
implot = plt.imshow(im, cmap=plt.cm.Greys_r)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.colorbar(implot, use_gridspec=True)
fig.tight_layout()
plt.title("Ajout de la barre des couleurs ")
pause()
plt.clf()  # nettoie la figure courante (retire la barre des couleurs)


# avec PIL
imPIL = PIL.Image.open('einstein.jpg')
imPIL.show()  # l'affichage est non bloquant, PIL sauve dans un fichier temporaire et utilise le viewer de l'OS
input("Press Enter to continue...")

# ----------------- rogner une  image -----------------

# avec Numpy
im = skimage.io.imread('einstein.jpg')
oeil = im[280:340, 290:390]
plt.imshow(oeil, cmap=plt.cm.Greys_r)
plt.title(u"Image Rognée ")
pause()

# avec PIL
imPIL = PIL.Image.open('einstein.jpg')
im2 = imPIL.crop((290, 280, 390, 340))
im2.size


# ----------------transformation spatiales----------------

# example d'animation avec matplotlib
im = cv2.imread('einstein.jpg', 0)
rows, cols = im.shape
imPlot = plt.imshow(im, cmap=plt.cm.Greys_r, animated=True,)
plt.show()
plt.draw()

# rotations avec OpenCV
plt.title(u"Rotation avec OpenCV ")
for theta in numpy.linspace(0, 360, 80):
    M = cv2.getRotationMatrix2D((cols/2, rows/2), theta, 1)
    dst = cv2.warpAffine(im, M, (cols, rows))
    imPlot.set_data(dst)
    plt.pause(0.0001)
    plt.draw()

# rotations avec scipy.ndimage
plt.title(u"Rotation avec scipy.ndimage")
for theta in numpy.linspace(0, 90, 20):
    dst = scipy.ndimage.rotate(im, theta, mode='constant')
    imPlot.set_data(dst)
    plt.pause(0.0001)
    plt.draw()

# rotations avec PIL
plt.title(u"Rotation avec PIL")
imPIL = PIL.Image.open('einstein.jpg')
for theta in numpy.linspace(0, 360, 80):
    print(theta)
    dst = imPIL.rotate(theta)
    imPlot.set_data(numpy.array(dst))
    plt.pause(0.0001)  # semble necessaire sous windows
    plt.draw()


# rotations avec scikit-image
plt.title(u"Rotation avec scikit-image")
for theta in numpy.linspace(0, 360, 80):
    tform = tf.SimilarityTransform(scale=1, rotation=theta*math.pi / 180,
                                   translation=(0, 0))
    # need to convert back to values between 0 and 255
    dst = (tf.warp(im, tform)*255).astype(numpy.uint8)
    imPlot.set_data(numpy.array(dst))
    plt.draw()


# --------------- seuillage -------------------------
# avec les operateurs de Numpy
im = cv2.imread('einstein.jpg', 0)
imBinaire = im < 120
plt.imshow(imBinaire, cmap=plt.cm.Greys_r)
pause()
im[imBinaire] = 0
plt.imshow(im, cmap=plt.cm.Greys_r)
plt.title(u"seuillage ")
pause()
# using opencv

# --------------- transformation des intensités-------------------------


def f(x):
    return int(pow(x/255.0, 3)*255)


im = cv2.imread('einstein.jpg', 0)

# avec numpy
# obtient une version de la fonction qui marche sur chacun des elements du tableau
fv = numpy.vectorize(f)
im2 = fv(im)
plt.title(u"mapping des instensités x->x^3 avec numpy")
plt.imshow(im2, cmap=plt.cm.Greys_r)
pause()

# avec PIL
imPIL = PIL.Image.open('einstein.jpg')
im2 = PIL.Image.eval(imPIL, f)
plt.title(u"mapping des instensités x->x^3 avec PIL")
plt.imshow(numpy.array(im2), cmap=plt.cm.Greys_r)
pause()
# -------- calcul de l'histogramme ---------------

# avec Numpy.histogram
im = skimage.io.imread('einstein.jpg')
h, bins = numpy.histogram(im, numpy.arange(0, 255))
hist_cum = numpy.cumsum(h)  # histograme cumulé
plt.subplot(1, 2, 1)
plt.plot(h)
plt.subplot(1, 2, 2)
plt.plot(hist_cum)
plt.title(u"histogramme et histogramme cummulé avec Numpy")
pause()
plt.clf()

# avec Numpy.bincount
h = numpy.bincount(im.ravel(), minlength=256)
plt.plot(h)
plt.title(u"histogramme avec Numpy.bincount")
pause()

# avec scikit-image
h, bins = skimage.exposure.histogram(im)
plt.plot(h)
pause()

# avec OpenCV
h = cv2.calcHist([im], [0], None, [256], [0, 256])
plt.plot(h)
plt.title(u"histogramme avec OpenCV")
pause()
# avec PIL
imPIL = PIL.Image.open('einstein.jpg')
h = imPIL.histogram()  # revoie une liste python
plt.plot(h)
plt.title(u"histogramme avec PIL")
pause()
plt.clf()

# ----------- egalisation d'histograme--------------

# avec scikit-image
im2 = skimage.exposure.equalize_hist(im)
plt.imshow(im2, cmap=plt.cm.Greys_r)
plt.title(u"Histogramme egalisé avec scikit-image")
pause()

# avec OpenCV
im2 = cv2.equalizeHist(im)
plt.imshow(im2, cmap=plt.cm.Greys_r)
plt.title(u"Histogramme egalisé avec OpenCV")
pause()

# avec PIL
imPIL = PIL.Image.open('einstein.jpg')
im2 = ImageOps.equalize(imPIL)  # ne marche pas sur ma machine..
plt.imshow(numpy.array(im2), cmap=plt.cm.Greys_r)
plt.title(u"Histogramme egalisé avec PIL")
pause()

# ------------ convolution---------------

# avec scipy.ndimage
weights = 0.5*numpy.array([-1, 0, 1]).reshape(3, 1)
dim_dx = scipy.ndimage.convolve(im.astype(float), weights)
# attention il faut utiliser un type qui peut prendre des valeurs négatives !!!
# d'ou l'utilisation de im.astype(float)
plt.imshow(dim_dx, cmap=plt.cm.Greys_r)
plt.title(u"Convolution avec scipy.ndimage")
pause()

# avec scikit-image
dim_dx = skimage.filters.edges.convolve(im.astype(int), weights)
plt.imshow(dim_dx, cmap=plt.cm.Greys_r)
plt.title(u"Convolution avec scikit-image")
pause()

# avec OpenCV

# il faut copier la mémoire sinon cela ne marche pas...
weightsFlipped2 = weights[::-1, ::-1].copy()
# filter 2D dans opencv fait une corrélation et non pas un convolution , il faut donc retourner le kernel
dim_dx = dst = cv2.filter2D(im.astype(float), -1, weightsFlipped2)
plt.imshow(dim_dx, cmap=plt.cm.Greys_r)
plt.title(u"Convolution avec OpenCV")
pause()

# avec PIL
imPIL = PIL.Image.open('einstein.jpg')
# limité à des noyaux de taille 3x3 ou 5x5
im2 = imPIL.filter(PIL.ImageFilter.Kernel(
    (3, 3), [0, 0, 0, -1, 0, 1, 0, 0, 0]))
plt.title(u"Convolution avec PIL")
pause()


# ----------filtres predefinis ----------------

# avec scipy.ndimage
im_smooth = scipy.ndimage.gaussian_filter(im, 4)


# ------------canny -------------

# avec scikit-image
edges = feature.canny(im, sigma=5)
plt.imshow(edges, cmap=plt.cm.Greys_r)
plt.title(u"Bords de Canny avec scikit-image")
pause()

# avec opencv
min_threshold = 50
max_threshold = 150
# besoin de filtrer un peu l'image d'abord
edges = cv2.Canny(cv2.GaussianBlur(im, (5, 5), 2),
                  min_threshold, max_threshold)
plt.imshow(edges, cmap=plt.cm.Greys_r)
plt.title(u"Bords de Canny avec OpenCV")
pause()
