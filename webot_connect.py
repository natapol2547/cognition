from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from controller import Camera, Robot

from utils.image import (
    gaussian_blur,
    image_to_binary,
    read_image,
    write_image,
    resize_image,
    sobel_filter,
)
from utils.robot_func import load_webots_robot_class
from utils.color_space import oklab_to_oklch, rgb_to_oklab, rgb_to_oklch
from utils.blob import blobize, filter_blobs_by_pixel_count, get_blob_average_color_oklab, get_blob_by_color, group_blobs, is_blob_moving
import time

def _find_gradient_oklch(image):
    blurred_image = gaussian_blur(image, 5)
    oklch_image = rgb_to_oklch(blurred_image)
    oklc_image = oklch_image[..., :2]
    oklc_image[..., 1] = oklc_image[..., 1] / 0.37 # Giving a gain for Chroma
    gradient_image = sobel_filter(oklc_image, 'gradient')
    return gradient_image

def find_gradient(image):
    blurred_image = gaussian_blur(image, 3)
    oklab_image = rgb_to_oklab(blurred_image)
    oklch_image = oklab_to_oklch(oklab_image)
    oklab_image[..., 1:] = oklab_image[..., 1:] * + 0.5 # Normalize for a* and b*
    oklabchroma_image = np.concatenate([oklab_image, oklch_image[..., 1:2]], axis=-1)
    oklabchroma_image[..., 3] = oklabchroma_image[..., 3] / 0.37 # Giving a gain for Chroma
    # Test
    oklabchroma_image = np.mean(oklabchroma_image, axis=-1)
    gradient_image = sobel_filter(oklabchroma_image, 'gradient')
    return gradient_image


def get_image(robot: Robot, width: int, height: int) -> np.ndarray | None:
    camera = cast("Camera", robot.getDevice("cam"))
    raw_image = camera.getImage()
    if raw_image:
        image_array = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        # Drop the Alpha channel
        image_array = image_array[:, :, :3]
        return image_array
    return None

def run_robot() -> None:
    RobotClass = load_webots_robot_class()
    robot = RobotClass()
    timestep = int(robot.getBasicTimeStep())

    previous_image = None

    camera = cast("Camera", robot.getDevice("cam"))
    camera.enable(timestep)
    
    width = camera.getWidth()
    height = camera.getHeight()

    print("Camera initialized. Press 'q' in the OpenCV window to exit.")

    while robot.step(timestep) != -1:
        image_array = get_image(robot, width, height)
        if image_array is None:
            continue
        # Flip the image vertically
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        if previous_image is None:
            # No previous image, so we can't compare
            previous_image = image_rgb
            continue
        gradient_image1 = find_gradient(previous_image)
        blobs1 = blobize(previous_image, gradient_image1, 0.2)
        blobs1 = filter_blobs_by_pixel_count(blobs1, 25)
        
        yellow_blob1 = get_blob_by_color(blobs1, (227, 212, 69), 0.1)
        goal_blob1 = get_blob_by_color(blobs1, (105, 212, 44), 0.1)
        
        # Image 2
        gradient_image2 = find_gradient(image_rgb)
        blobs2 = blobize(image_rgb, gradient_image2, 0.2, debug=True)
        blobs2 = filter_blobs_by_pixel_count(blobs2, 25)
        groups = group_blobs(blobs1, blobs2)
        
        yellow_blob2 = get_blob_by_color(blobs2, (227, 212, 69), 0.1)
        goal_blob2 = get_blob_by_color(blobs2, (105, 212, 44), 0.1)
        
        if yellow_blob1 and yellow_blob2:
            if is_blob_moving(yellow_blob1, yellow_blob2):
                print("Yellow blob is moving")
            else:
                print("Yellow blob is not moving")
        else:
            print("Yellow blob not found")
        if goal_blob1 and goal_blob2:
            if is_blob_moving(goal_blob1, goal_blob2):
                print("Goal blob is moving")
            else:
                print("Goal blob is not moving")
        else:
            print("Goal blob not found")
        # Remove images from outputs folder
        for file in os.listdir("outputs"):
            os.remove(os.path.join("outputs", file))
        for i, group in enumerate(groups):
            write_image(f"outputs/group_{i}.png", group[0])
            write_image(f"outputs/group_{i}_pair.png", group[1])    
            
        previous_image = image_rgb
        cv2.imshow("Robot Vision", image_array)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_robot()