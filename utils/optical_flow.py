import numpy as np
from .image import is_image_rgb, gaussian_blur, rgb_to_gray, convolution, SOBEL_KERNEL_X, SOBEL_KERNEL_Y, resize_bilinear
import matplotlib.pyplot as plt


def pad_image(img, kernel_size):
    # Pad the image
    pad_h = kernel_size//2
    pad_w = kernel_size//2

    if is_image_rgb(img):
        output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
        padded_image = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
        
        # Convolve for color image
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    cropped_image = padded_image[i:i+kernel_size, j:j+kernel_size, k]
                    output[i, j, k] = np.sum(cropped_image * kernel_size)
    else:
        output = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        padded_image = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    return padded_image

def image_to_binary32(image, threshold=0.5):
    image = image / 255.0
    return (image > threshold).astype(np.float32) * 255.0

def optical_flow_vector_field(img_1, img_2, kernel_size=5, blur_iterations=1, eig_thresh = 1e-4):
    if img_1.shape != img_2.shape:
        raise ValueError("Input images must have the same dimensions.")    
    
    if img_1.dtype != img_2.dtype:
        raise ValueError("Input images must have the same data type.")
    
    if blur_iterations > 0:
        img_1 = gaussian_blur(img_1, iterations=blur_iterations)
        img_2 = gaussian_blur(img_2, iterations=blur_iterations)

    if is_image_rgb(img_1):
        img_1 = rgb_to_gray(img_1)
    if is_image_rgb(img_2):
        img_2 = rgb_to_gray(img_2)

    h, w = img_1.shape[0], img_1.shape[1]
    optical_flow_vector_field = np.zeros((h, w, 2, 1), dtype=np.float32)

    pad_h = kernel_size//2
    pad_w = kernel_size//2
    
    padded_image_1 = pad_image(img_1, kernel_size)
    padded_image_2 = pad_image(img_2, kernel_size)
    
    Ix_array = convolution(padded_image_1, SOBEL_KERNEL_X)
    Iy_array = convolution(padded_image_1, SOBEL_KERNEL_Y)

    It_array = padded_image_2 - padded_image_1

    for i in range(optical_flow_vector_field.shape[0]):
        for j in range(optical_flow_vector_field.shape[1]):
            
            Ix = Ix_array[i+pad_h:i+pad_h+kernel_size, j+pad_w:j+pad_w+kernel_size]
            Iy = Iy_array[i+pad_h:i+pad_h+kernel_size, j+pad_w:j+pad_w+kernel_size]
            It = It_array[i+pad_h:i+pad_h+kernel_size, j+pad_w:j+pad_w+kernel_size]
            
            Ix = Ix.ravel()
            Iy = Iy.ravel()
            
            It = It.ravel()
            It = It[:,np.newaxis]

            A = np.stack((Ix, Iy), axis=1)
            b = -It

            G = A.T @ A                          
            eigvals = np.linalg.eigvalsh(G)    
                         
            if eigvals[0] < eig_thresh:
                uv = np.zeros((2, 1), dtype=np.float32)
            else:
                uv = np.linalg.lstsq(A, b, rcond=None)[0]
                if np.isnan(uv).any():
                    uv = np.zeros((2, 1), dtype=np.float32)
                    
                optical_flow_vector_field[i, j] = uv

    return optical_flow_vector_field

def optical_flow_vector(optical_flow_vector_field):
    flow_vector = np.mean(optical_flow_vector_field, axis=(0, 1))
    return flow_vector

def optical_flow_magnitude(flow_vector_field):
    magnitude = np.sqrt(flow_vector_field[..., 0, 0]**2 + flow_vector_field[..., 1, 0]**2)
    return magnitude

def optical_flow_pyramid(img_1, img_2, levels=3, initial_kernel_size=3,res=(100,75) ,blur_iterations=1, eig_thresh = 1e-2):

    img_1 = resize_bilinear(img_1, res[0], res[1])
    img_2 = resize_bilinear(img_2, res[0], res[1])

    h, w = img_1.shape[0], img_1.shape[1]
    flow_pyramid = np.zeros((h, w, 2, 1), dtype=np.float32)

    kernel_size = initial_kernel_size
    for i in range(levels):
        flow_field = optical_flow_vector_field(img_1, img_2, kernel_size=kernel_size, blur_iterations=blur_iterations, eig_thresh = eig_thresh)
        flow_pyramid += flow_field
        kernel_size += 2
        
    return flow_pyramid / levels

def plot_optical_flow(flow_field, step=8, scale=1.0, background=None, output_path=None):
    # remove useless last axis if present
    if flow_field.ndim == 4:
        flow_field = flow_field[..., 0]   # -> (H, W, 2)

    U = flow_field[..., 0]
    V = flow_field[..., 1]

    H, W = U.shape

    # subsample (important for speed + readability)
    U = U[::step, ::step]
    V = V[::step, ::step]
    V = -V  # Flip V to match array indexing (row increases downward)

    x = np.arange(0, W, step)
    y = np.arange(0, H, step)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(6, 6))

    if background is not None:
        plt.imshow(background, cmap='gray')

    plt.quiver(
        X, Y,
        U, V,
        angles='xy',
        scale_units='xy',
        scale=scale
    )

    plt.axis('off')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
