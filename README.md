# face-roi

Project to find the regions of interest in the face and mask them from a stream of images / video.

The project is currently setup to find the ROIs forehead, left cheek and right cheek,
to change the ROIs use the reference of the [Canonical face model image](canonical_face_model_uv_viz.png) which is taken from the mediapipe [repo](https://github.com/google-ai-edge/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png).

## Dependencies

Depends on numpy, opencv, imageio, h5py and [mediapipe](https://github.com/google-ai-edge/mediapipe).

```bash
pip install numpy opencv-python mediapipe imageio h5py
```

## Usage

Use either the get_face_roi function or draw_roi functions from face_roi.py.

```python
from face_roi import display_roi, get_face_roi, mat_loader, mat_converter
```

Also note that using mat_converter and display_roi functions is not working as expected. It is reccomended to use something like converter.py to convert the mat files beforehand.

main.py works, but it is just a proof of concept with little functionality.


## Examples

Examples are provided in the [examples notebook](example.ipynb).
