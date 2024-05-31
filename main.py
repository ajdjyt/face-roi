import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

det_conf=0.5
track_conf=0.5
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

forehead_outline = [21, 71, 70, 46, 225, 224, 223, 222, 221, 193, 168, 417, 441, 442, 443, 444, 445, 276, 300, 301, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54]
forehead_inners = [68, 104, 69, 108, 151, 337, 299, 333, 298] + [43, 105, 66, 107, 9, 336, 296, 334, 293] + [53, 52, 65, 55, 8, 285, 295, 282, 283]
forehead_landmarks = forehead_outline + forehead_inners

left_cheek_outline = [234, 227, 116, 117, 118, 101, 36, 203, 165, 92, 186, 57, 43 ] + [202, 210, 169] + [150, 136, 172, 58, 132, 93]

right_cheek_outline = [454, 447, 345, 346, 330, 266, 423, 391, 322, 410, 287, 273 ] + [422, 430, 394] + [379, 365, 397, 288, 361, 323]


outlines = [forehead_outline, left_cheek_outline, right_cheek_outline]

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=det_conf,
    min_tracking_confidence=track_conf) as face_mesh:
  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Process Image with face_mesh
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Display results 
    if results.multi_face_landmarks:
        mask = np.zeros_like(image)

        for face_landmarks in results.multi_face_landmarks:
          
            selected_landmarks = []
            normalized_landmark_list = landmark_pb2.NormalizedLandmarkList()
            
            for idx in forehead_landmarks:
                # Create list of landmarks
                selected_landmarks.append(face_landmarks.landmark[idx])
                # Create LL of landmarks
                landmark = landmark_pb2.NormalizedLandmark()
                landmark.x = face_landmarks.landmark[idx].x
                landmark.y = face_landmarks.landmark[idx].y
                normalized_landmark_list.landmark.append(landmark)
            
            # Plot the outline on mask for every ROI
            for selected_outline in outlines:
              selected_points = np.array([(int(face_landmarks.landmark[landmark].x * image.shape[1]), int(face_landmarks.landmark[landmark].y * image.shape[0])) for landmark in selected_outline])
              cv2.fillPoly(mask, [selected_points], (255, 255, 255))
            
            # Plot the mesh landmarks in every ROI
            for landmark in selected_landmarks:
              cx, cy = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
              cv2.circle(image, (cx, cy), 1, (250, 200, 200), -1)
        
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, mask)
        cv2.imshow('Masked Face', masked_image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
  
cap.release()
cv2.destroyAllWindows()
