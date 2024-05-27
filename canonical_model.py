import numpy as np
import cv2
import math
def load_obj_file(file_path):
    uv_coords = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('vt '):
                uv = line.strip().split()[1:]
                uv = [float(coord) for coord in uv]
                uv_coords.append(uv)
    uv_coords = np.array(uv_coords)
    return uv_coords
obj_path = 'canonical_face_model.obj'
uv_coords = load_obj_file(obj_path)
white_img = np.ones((2048,2048 , 3), dtype=np.uint8) * 255
img_h, img_w = white_img.shape[:2]
uv_coords[:, 1] = 1 - uv_coords[:, 1]
uv_coords[:, 0] = uv_coords[:, 0] * white_img.shape[1]
uv_coords[:, 1] = uv_coords[:, 1] * white_img.shape[0]
for num, uv in enumerate(uv_coords):
    cv2.circle(white_img, (int(uv[0]), int(uv[1])), 1, (0, 255, 0), -1)
    cv2.putText(white_img, str(num), (int(uv[0]), int(uv[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
cv2.imshow('canonical_face_model', white_img)
cv2.imwrite('canonical_face_model.png', white_img)