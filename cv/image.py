from typing import Literal
import numpy as np
import skimage as sk

SOBEL_KERNEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
SOBEL_KERNEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

def is_image_rgb(image):
    return image.ndim == 3

def rgb_to_gray(image):
    if not is_image_rgb(image):
        return image
    image = image.astype(np.float64) / 255
    dot_product = np.dot(image, [0.299, 0.587, 0.114])
    return (dot_product * 255).astype(np.uint8)

def read_image(path):
    return sk.io.imread(path)

def write_image(path, image):
    sk.io.imsave(path, image.astype(np.uint8), check_contrast=False)

def resize_image(image, new_w, new_h):
    old_h, old_w = image.shape[:2]
    
    if is_image_rgb(image):
        output = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    else:
        output = np.zeros((new_h, new_w), dtype=image.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            src_i = i * old_h // new_h
            src_j = j * old_w // new_w
            output[i, j] = image[src_i, src_j]
    
    return output

def resize_bilinear(image, new_w, new_h):
    old_h, old_w = image.shape[:2]
    
    if is_image_rgb(image):
        output = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    else:
        output = np.zeros((new_h, new_w), dtype=image.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            src_i = i * old_h / new_h
            src_j = j * old_w / new_w
            
            i0 = int(np.floor(src_i))
            i1 = min(i0 + 1, old_h - 1)
            j0 = int(np.floor(src_j))
            j1 = min(j0 + 1, old_w - 1)
            
            di = src_i - i0
            dj = src_j - j0
            
            if is_image_rgb(image):
                top_left = image[i0, j0]
                top_right = image[i0, j1]
                bottom_left = image[i1, j0]
                bottom_right = image[i1, j1]
                
                top = (1 - dj) * top_left + dj * top_right
                bottom = (1 - dj) * bottom_left + dj * bottom_right
                output[i, j] = (1 - di) * top + di * bottom
            else:
                top_left = image[i0, j0]
                top_right = image[i0, j1]
                bottom_left = image[i1, j0]
                bottom_right = image[i1, j1]
                
                top = (1 - dj) * top_left + dj * top_right
                bottom = (1 - dj) * bottom_left + dj * bottom_right
                output[i, j] = (1 - di) * top + di * bottom
    
    return output

def resize_uniform(image, ratio):
    new_h = int(image.shape[0] * ratio)
    new_w = int(image.shape[1] * ratio)
    return resize_image(image, new_h, new_w)

def convolution(image, kernel):
    # Pad the image
    pad_h = kernel.shape[0]//2
    pad_w = kernel.shape[1]//2
    
    if is_image_rgb(image):
        output = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
        
        # Convolve for color image
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    cropped_image = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1], k]
                    output[i, j, k] = np.sum(cropped_image * kernel)
    else:
        output = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Convolve for grayscale image
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                cropped_image = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                output[i, j] = np.sum(cropped_image * kernel)
    
    return output

# https://stackoverflow.com/questions/74343085/how-do-i-write-code-for-a-2d-gaussian-kernel
def create_gaussian_kernel(size, sigma=1):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

GAUSSIAN_KERNEL = create_gaussian_kernel(3, 1)

def gaussian_blur(image, iterations=1):
    for _ in range(iterations):
        image = convolution(image, GAUSSIAN_KERNEL)
    return image

def sobel_filter(image, mode: Literal['magnitude', 'maximum', 'gradient'] = 'magnitude'):
    if image.dtype == 'uint8':
        image = image.astype(np.float64) / 255
    if is_image_rgb(image):
        gray = np.mean(image, axis=-1)
    else:
        gray = image
    sobel_x = convolution(gray, SOBEL_KERNEL_X)
    sobel_y = convolution(gray, SOBEL_KERNEL_Y)
    stack = np.stack([sobel_x, sobel_y], axis=-1)
    if mode == 'magnitude':
        magnitude = np.sqrt(np.sum(stack**2, axis=-1))
        return (magnitude * 255).astype(np.uint8)
    elif mode == 'maximum':
        # Maximum magnitude across color channels
        return (np.max(stack, axis=-1) * 255).astype(np.uint8)
    elif mode == 'gradient':
        return stack

def image_to_binary(image, threshold=0.5):
    if image.dtype == 'uint8':
        image = image.astype(np.float64) / 255
    return (image > threshold).astype(np.uint8) * 255

def anisotropic_gaussian_kernel(res, sigma_x=None, sigma_y=None):
    A = 1
    x0 = res[1] // 2
    y0 = res[0] // 2

    if sigma_x is None:
        sigma_x = x0 / 4
    if sigma_y is None:
        sigma_y = y0 / 4

    filter = np.zeros_like(np.zeros(res), dtype=np.float64)
    for x in range(filter.shape[1]):
        for y in range(filter.shape[0]):
            filter[y, x] = A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))
    
    return filter

def apply_equivalent_filter(img,filter):
    if img.shape != filter.shape:
        raise ValueError("Image and filter must have the same shape, but got {} and {}".format(img.shape, filter.shape))
    
    output = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            output[j, i] = img[j, i] * filter[j, i]
    return output


# if __name__ == "__main__":
#     img = read_image("Depth_camera_outputs/depth_camera_image_3.jpg")
#     gray = rgb_to_gray(img)
#     filter = anisotropic_gaussian_kernel(gray)
#     write_image("anisotropic_gaussian_filter.jpg", filter / np.max(filter) * 255)
#     output = apply_equivalent_filter(gray, filter)
#     write_image("equivalent_filter_output.jpg", output)
