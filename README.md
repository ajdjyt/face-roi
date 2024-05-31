# face-roi

Project to find the regions of interest in the face and mask them from a stream of images / video.

The project is currently setup to find the ROIs forehead, left cheek and right cheek, 
to change the ROIs use the reference of the [Canonical face model image](canonical_face_model_uv_viz.png).

## Dependencies

Depends on numpy, opencv and [mediapipe](https://github.com/google-ai-edge/mediapipe)

```bash
pip install numpy opencv-python mediapipe
```
