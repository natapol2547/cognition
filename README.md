# Cognition - Robot Vision & Object Tracking System

A computer vision system for robot perception that performs real-time blob detection and temporal object tracking using perceptually uniform color spaces and gradient-based edge detection.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COGNITION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  Image       │───▶│  Gradient    │───▶│    Blob      │───▶│  Temporal │  │
│  │  Acquisition │    │  Detection   │    │  Extraction  │    │  Tracking │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘  │
│        │                   │                    │                   │       │
│        ▼                   ▼                    ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ Webots Camera│    │ Gaussian Blur│    │ Flood-Fill   │    │ Histogram │  │
│  │ or Static    │    │ RGB → OKLCH  │    │ Segmentation │    │ Matching  │  │
│  │ Images       │    │ Sobel Filter │    │ + Histograms │    │ Algorithm │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
cognition/
├── main.py              # Standalone image processing demo
├── webot_connect.py     # Webots robot controller integration
├── utils/
│   ├── image.py         # Image I/O, convolution, filtering
│   ├── color_space.py   # OKLAB/OKLCH color space conversions
│   └── blob.py          # Blob detection and tracking
└── outputs/             # Generated visualization outputs
```

---

## Core Components

### 1. Image Processing (`utils/image.py`)

Low-level image manipulation primitives implemented from scratch using NumPy.

#### Functions

| Function | Description |
|----------|-------------|
| `read_image(path)` | Load image from disk using scikit-image |
| `write_image(path, image)` | Save image to disk |
| `resize_image(image, new_w, new_h)` | Nearest-neighbor resize |
| `convolution(image, kernel)` | 2D convolution with edge padding |
| `gaussian_blur(image, iterations)` | Multi-pass Gaussian smoothing (3×3 kernel) |
| `sobel_filter(image, mode)` | Edge detection with gradient output |
| `image_to_binary(image, threshold)` | Threshold-based binarization |

#### Sobel Edge Detection

The Sobel filter computes image gradients using two 3×3 kernels:

```
Sobel X:           Sobel Y:
[-1  0  1]         [-1 -2 -1]
[-2  0  2]         [ 0  0  0]
[-1  0  1]         [ 1  2  1]
```

Supports three output modes:
- `'magnitude'` - Gradient magnitude as grayscale
- `'maximum'` - Max response across channels
- `'gradient'` - Raw (Gx, Gy) vector field for direction analysis

---

### 2. Color Space Conversion (`utils/color_space.py`)

Implements perceptually uniform **OKLAB** and **OKLCH** color spaces for improved edge detection and color comparison.

#### Why OKLCH?

Standard RGB/HSV color spaces have non-uniform perceptual distances. OKLAB provides:
- **Perceptual uniformity**: Equal numerical differences = equal perceived differences
- **Better edge detection**: Gradients in OKLCH correlate with human-perceived edges
- **Hue independence**: Lightness (L) and Chroma (C) are separated from Hue (H)

#### Conversion Pipeline

```
RGB (0-255) → Linear RGB → LMS → OKLAB (L, a, b) → OKLCH (L, C, H)
```

#### Key Functions

```python
rgb_to_oklab(rgb)      # RGB to OKLAB (Lightness, a*, b*)
oklab_to_oklch(lab)    # OKLAB to OKLCH (Lightness, Chroma, Hue)
rgb_to_oklch(rgb)      # Direct RGB to OKLCH conversion
oklch_to_rgb(lch)      # Inverse: OKLCH back to RGB
```

---

### 3. Blob Detection & Tracking (`utils/blob.py`)

The core perception module that segments images into distinct regions (blobs) and tracks them across frames.

#### Blob Data Structure

```python
@dataclass
class Blob:
    blob_image: np.ndarray           # RGBA mask of the blob region
    histrogram: np.ndarray           # 8×8×8 RGB color histogram
    gradient_histrogram: np.ndarray  # 8-bin gradient direction histogram
    center: tuple[float, float]      # Centroid (x, y) of the blob
```

#### Blob Extraction Algorithm (`blobize`)

1. **Gradient Computation**: Calculate edge magnitude from OKLCH-space gradients
2. **Edge Binarization**: Threshold gradient to create edge map
3. **Flood-Fill Segmentation**: Starting from non-edge pixels, expand regions using 4-connectivity
4. **Feature Extraction**: For each blob, compute:
   - Color histogram (8×8×8 bins in RGB space)
   - Gradient direction histogram (8 bins, 45° each)
   - Centroid position

```python
def blobize(image, gradient_image, threshold=0.1) -> list[Blob]:
    # 1. Create binary edge map from gradient magnitude
    magnitude = np.sqrt(np.sum(gradient_image**2, axis=-1))
    edge_image = image_to_binary(magnitude, threshold)
    
    # 2. Flood-fill from each unvisited non-edge pixel
    for each unvisited pixel (i, j):
        if not edge_pixel:
            # Expand region using stack-based flood fill
            # Build histograms and blob mask simultaneously
```

#### Temporal Blob Matching (`group_blobs`)

Matches blobs between consecutive frames using a weighted distance metric:

```
Distance = k1 × histogram_distance + k2 × center_distance + k3 × gradient_histogram_distance
```

Where:
- `histogram_distance`: Bhattacharyya-like distance between color histograms
- `center_distance`: Euclidean distance between blob centroids
- `gradient_histogram_distance`: Similarity of edge orientations

Default weights: `k1=1, k2=0.1, k3=1`

---

## Simulation Code

### Standalone Demo (`main.py`)

Processes two static images and outputs matched blob pairs:

```python
def find_gradient(image):
    """Compute perceptually-weighted gradient image."""
    blurred_image = gaussian_blur(image, 5)
    oklch_image = rgb_to_oklch(blurred_image)
    oklc_image = oklch_image[..., :2]  # Use only L and C channels
    oklc_image[..., 1] = oklc_image[..., 1] / 0.37  # Chroma gain
    gradient_image = sobel_filter(oklc_image, 'gradient')
    return gradient_image

def main():
    # Process frame 1
    image1 = read_image("frame1.jpg")
    image1 = resize_image(image1, 100, 75)
    gradient_image1 = find_gradient(image1)
    blobs1 = blobize(image1, gradient_image1)
    
    # Process frame 2
    image2 = read_image("frame2.jpg")
    image2 = resize_image(image2, 100, 75)
    gradient_image2 = find_gradient(image2)
    blobs2 = blobize(image2, gradient_image2)
    
    # Match blobs across frames
    groups = group_blobs(blobs1, blobs2)
    
    # Output matched pairs
    for i, group in enumerate(groups):
        write_image(f"outputs/group_{i}.png", group[0])
        write_image(f"outputs/group_{i}_pair.png", group[1])
```

---

### Webots Robot Controller (`webot_connect.py`)

Real-time robot vision integration with the Webots simulator.

#### Setup

Requires Webots installation with `WEBOTS_HOME` environment variable:

```bash
# .env file
WEBOTS_HOME=/path/to/webots
```

#### Robot Control Loop

```python
def run_robot():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # Initialize camera
    camera = robot.getDevice("cam")
    camera.enable(timestep)
    
    previous_image = None
    
    while robot.step(timestep) != -1:
        # Acquire current frame
        image_array = get_image(robot, width, height)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        if previous_image is None:
            previous_image = image_rgb
            continue
        
        # Process previous frame
        gradient_image1 = find_gradient(previous_image)
        blobs1 = blobize(previous_image, gradient_image1, 0.3)
        blobs1 = filter_blobs_by_pixel_count(blobs1, 25)  # Remove noise
        
        # Process current frame
        gradient_image2 = find_gradient(image_rgb)
        blobs2 = blobize(image_rgb, gradient_image2, 0.3)
        blobs2 = filter_blobs_by_pixel_count(blobs2, 25)
        
        # Track objects across frames
        groups = group_blobs(blobs1, blobs2)
        
        # Output visualizations
        for i, group in enumerate(groups):
            write_image(f"outputs/group_{i}.png", group[0])
            write_image(f"outputs/group_{i}_pair.png", group[1])
        
        previous_image = image_rgb
```

#### Camera Image Acquisition

```python
def get_image(robot, width, height):
    """Capture frame from Webots camera device."""
    camera = robot.getDevice("cam")
    raw_image = camera.getImage()
    if raw_image:
        # Convert raw bytes to numpy array (BGRA format)
        image_array = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        # Drop alpha channel
        return image_array[:, :, :3]
    return None
```

---

## Processing Pipeline Summary

```
Frame N-1                          Frame N
    │                                  │
    ▼                                  ▼
┌─────────┐                      ┌─────────┐
│Gaussian │                      │Gaussian │
│Blur (5x)│                      │Blur (5x)│
└────┬────┘                      └────┬────┘
     ▼                                ▼
┌─────────┐                      ┌─────────┐
│RGB→OKLCH│                      │RGB→OKLCH│
└────┬────┘                      └────┬────┘
     ▼                                ▼
┌─────────┐                      ┌─────────┐
│ Sobel   │                      │ Sobel   │
│Gradient │                      │Gradient │
└────┬────┘                      └────┬────┘
     ▼                                ▼
┌─────────┐                      ┌─────────┐
│Blobize  │                      │Blobize  │
│(segment)│                      │(segment)│
└────┬────┘                      └────┬────┘
     │                                │
     └──────────┬─────────────────────┘
                ▼
         ┌───────────┐
         │group_blobs│
         │(matching) │
         └─────┬─────┘
               ▼
        Matched Blob Pairs
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/edge_image.png` | Raw gradient magnitude |
| `outputs/edge_image_binary.png` | Thresholded edge map |
| `outputs/gradient_image1.png` | Gradient visualization |
| `outputs/group_N.png` | Blob N from frame 1 |
| `outputs/group_N_pair.png` | Matched blob N from frame 2 |

---

## Dependencies

- `numpy` - Array operations
- `scikit-image` - Image I/O
- `opencv-python` - Display and color conversion (Webots mode)
- `python-dotenv` - Environment configuration (Webots mode)

---

## Usage

### Static Image Processing

```bash
python main.py
```

### Webots Robot Simulation

1. Set up `.env` with `WEBOTS_HOME`
2. Configure Webots world with robot containing a camera device named `"cam"`
3. Run:

```bash
python webot_connect.py
```

---

## Key Design Decisions

1. **OKLCH Color Space**: Provides perceptually uniform edge detection, improving segmentation quality compared to RGB-based methods.

2. **Histogram-Based Matching**: Color and gradient histograms are robust to small translations and rotations, enabling reliable tracking.

3. **Flood-Fill Segmentation**: Simple but effective for real-time blob extraction without requiring learned models.

4. **Stack-Based Recursion**: Avoids Python's recursion limit for large connected regions.

5. **Minimum Pixel Filtering**: Removes noise blobs in robot vision mode (`filter_blobs_by_pixel_count`).
