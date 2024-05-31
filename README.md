# face-roi

Project to find the regions of interest in the face and mask them from a stream of images / video.

The project is currently setup to find the ROIs forehead, left cheek and right cheek, 
to change the ROIs use the reference of the [Canonical face model image](canonical_face_model_uv_viz.png) which is taken from the mediapipe [repo](https://github.com/google-ai-edge/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png).

## Dependencies

Depends on numpy, opencv and [mediapipe](https://github.com/google-ai-edge/mediapipe).

```bash
pip install numpy opencv-python mediapipe
```

## Running

Use main.py

```bash
python main.py
```