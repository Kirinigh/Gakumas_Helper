"""Shared, observed checkbox glyph and plate for the calibrated 480 profile."""
import cv2
import numpy as np


def read_checkbox(image, box, *, selected_area_max=60):
    if (getattr(image, 'shape', ()) != (1280, 720, 3)
            or not isinstance(box, tuple) or len(box) != 4
            or any(type(value) is not int for value in box)
            or min(box[:2]) < 0 or box[2:] != (30, 30)
            or box[0] + 30 > 720 or box[1] + 30 > 1280):
        raise ValueError('current checkbox geometry is unresolved')
    x, y, width, height = box
    crop = image[y:y + height, x:x + width]
    _, _, stats, _ = cv2.connectedComponentsWithStats((crop.min(2) > 245).astype('uint8'))
    checked = [row for row in stats[1:] if 3 <= row[0] <= 5 and 6 <= row[1] <= 8
        and 19 <= row[2] <= 21 and 14 <= row[3] <= 16]
    if len(checked) == 1:
        plate = crop[6, 6].astype('int16')
        if np.abs(plate - [203, 203, 200]).max() <= 5 and 70 <= checked[0][4] <= 85:
            return False
        if np.abs(plate - [0, 118, 255]).max() <= 5 and 50 <= checked[0][4] <= selected_area_max:
            return True
    raise ValueError('current checkbox state is unresolved')
