from matplotlib import pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import pydicom as dicom
from skimage import exposure, img_as_ubyte, color
from skimage.transform import rescale, resize, downscale_local_mean
import concurrent.futures


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h * w, 3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
                                              clusters,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                              rounds,
                                              cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


def process(path):
    path_in_str = str(path)
    print('|', end='')

    src = cv2.imread(path_in_str)

    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    # a = cv2.equalizeHist(a)
    # b = cv2.equalizeHist(b)
    lab = cv2.merge((l2, a, b))  # merge channels
    src = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    bsrc = cv2.GaussianBlur(src, (9, 9), 0)
    sobelx = cv2.Sobel(bsrc, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(bsrc, cv2.CV_64F, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    src = cv2.resize(src, (1024, 1024), interpolation=cv2.INTER_AREA)
    reduced = kmeans_color_quantization(src, clusters=32)
    laplacian = cv2.Laplacian(reduced, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    laplacian = cv2.resize(laplacian, (512, 512), interpolation=cv2.INTER_AREA)
    laplacian = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.equalizeHist(laplacian)

    src = cv2.resize(src, (512, 512), interpolation=cv2.INTER_AREA)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = cv2.equalizeHist(src)

    grad = cv2.resize(grad, (512, 512), interpolation=cv2.INTER_AREA)
    grad = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    grad = cv2.equalizeHist(grad)

    dest = cv2.merge((src, laplacian, grad))
    cv2.imwrite(path_in_str, dest)
    return path_in_str


# Create a pool of processes. By default, one is created for each CPU in your machine.
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Get a list of files to process
    pathlist = Path('/headless/data/mel/pytest2/').glob('**/*.jp*')
    i = 0
    # Process the list of files, but split the work across the process pool to use all CPUs!
    for image_file, thumbnail_file in zip(pathlist, executor.map(process, pathlist)):
        if i % 1000 == 0:
            print(str(i))
        i = i + 1

print("done.")