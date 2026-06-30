import cv2
import numpy as np


def rgb_to_hsv(r: int, g: int, b: int) -> tuple[int, int, int]:
    """
    将 RGB 转换为 OpenCV HSV。
    
    输入:
        r, g, b: 0-255
    
    输出:
        h, s, v:
            h: 0-179
            s: 0-255
            v: 0-255
    """
    rgb_pixel = np.uint8([[[r, g, b]]])

    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)

    h, s, v = hsv_pixel[0][0]

    return int(h), int(s), int(v)


if __name__ == "__main__":

    color = (  121, 174, 173   )

    h, s, v = rgb_to_hsv(color[0], color[1], color[2])

    print(f"RGB({color}) -> HSV({h}, {s}, {v})")