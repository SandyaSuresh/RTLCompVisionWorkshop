from cv2.dnn import imagesFromBlob
import cv2
import numpy as np
import matplotlib.pyplot as plt

filter_name = "gaussian"
img = cv2.imread("rose.jpeg")
# img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
"""
Apply different convolution kernels or OpenCV filters 
based on filter_name using a match-case switch.
"""

match filter_name.lower():

    # ------------------------ Gaussian Blur ------------------------
    case "gaussian":
        kernel_gaussian_blur =np.array([[1, 4, 1],
                                      [4, 6, 4],
                                      [1, 4, 1]], dtype=np.float32)
        normalized = kernel_gaussian_blur/kernel_gaussian_blur.sum()
        filtered =  cv2.filter2D(img, -1, normalized)
    # ------------------------ Prewitt Filters ----------------------
    case "prewitt_x":
        kernel_prewitt_x = np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]], dtype=np.float32)
        filtered =  cv2.filter2D(img, -1, kernel_prewitt_x)

    case "prewitt_y":
        kernel_prewitt_y = np.array([[-1, -1, -1],
                                      [ 0,  0,  0],
                                      [ 1,  1,  1]], dtype=np.float32)
        filtered = cv2.filter2D(img, -1, kernel_prewitt_y)

    # ------------------------ Laplacian Kernel ---------------------
    case "laplacian":
        # using a classic 3x3 Laplacian kernel
        kernel_lap = np.array([[0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]], dtype=np.float32)
        filtered = cv2.filter2D(img, -1, kernel_lap)

    # ------------------------ Sharpening Kernel --------------------
    case "sharpen":
        kernel_sharp = np.array([[ 0, -1,  0],
                                  [-1,  5, -1],
                                  [ 0, -1,  0]], dtype=np.float32)
        filtered = cv2.filter2D(img, -1, kernel_sharp)

    # ------------------------ Bilateral Filter ---------------------
    case "bilateral":
        # d = neighborhood size, sigmaColor, sigmaSpace
        filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # ------------------------ Invalid ------------------------------
    case _:
        custom = np.array([[ 0, -1,  0],
                                  [-1,  5, -1],
                                  [ 0, -1,  0]], dtype=np.float32)
        filtered = cv2.filter2D(img, -1, custom)

#Google Colab doesn't support openCV's imshow. Natively, the command would be cv2.imshow(filtered)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(filtered)
plt.title("Filtered")
plt.axis("off")
plt.show()