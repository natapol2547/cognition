from cv.image import (
    gaussian_blur,
    read_image,
    write_image,
    resize_image,
    sobel_filter,
)
from cv.color_space import rgb_to_oklab, rgb_to_oklch
from cv.blob import blobize, group_blobs

import numpy as np
import time

DEBUG = True

def find_gradient(image):
    blurred_image = gaussian_blur(image, 5)
    write_image("outputs/blurred_image.png", blurred_image)
    oklab_image = rgb_to_oklab(blurred_image)
    oklch_image = rgb_to_oklch(blurred_image)
    oklab_image[..., 1:] = oklab_image[..., 1:] * + 0.5 # Normalize for a* and b*
    oklabchroma_image = np.concatenate([oklab_image, oklch_image[..., 1:2]], axis=-1)
    oklabchroma_image[..., 3] = oklabchroma_image[..., 3] / 0.37 # Giving a gain for Chroma
    gradient_image = sobel_filter(oklabchroma_image, 'gradient')
    write_image("outputs/gradient_image.png", gradient_image)
    return gradient_image

def main():
    start_time = time.time()
    # Image 1
    image1 = read_image("frame1.jpg")
    image1 = resize_image(image1, 100, 75)
    gradient_image1 = find_gradient(image1)
    if DEBUG:
        write_image("outputs/gradient_image1.png", gradient_image1)
    blobs1 = blobize(image1, gradient_image1, debug=DEBUG)
    
    # Image 2
    image2 = read_image("frame2.jpg")
    image2 = resize_image(image2, 100, 75)
    gradient_image2 = find_gradient(image2)
    blobs2 = blobize(image2, gradient_image2)
    groups = group_blobs(blobs1, blobs2)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    for i, group in enumerate(groups):
        write_image(f"outputs/group_{i}.png", group[0])
        write_image(f"outputs/group_{i}_pair.png", group[1])
if __name__ == "__main__":
    main()
