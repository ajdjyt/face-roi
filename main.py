import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

det_conf=0.5
track_conf=0.5

#cheek_left_indices = [146, 150, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 130, 131, 132, 133, 134, 135]
#cheek_right_indices = [330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 263, 262, 261, 260, 259, 258]
#forehead_indices = [10, 272, 276, 280, 283, 285, 290, 293, 296, 168, 191]

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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
      for face_landmarks in results.multi_face_landmarks:
        '''
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        '''
        left_cheek_landmarks = [269, 150]
        right_cheek_landmarks = [47, 165]
        forehead_landmarks = [353, 134, 348]
        #left_cheek_landmarks = [234, 291, 5, 6, 7, 8, 269, 150]
        #right_cheek_landmarks = [454, 268, 271, 270, 265, 47, 165]
        #forehead_landmarks = [10, 151, 171, 148, 152, 377, 378, 379, 380, 381]

        selected_landmarks = []
        for idx in left_cheek_landmarks + right_cheek_landmarks + forehead_landmarks:
          selected_landmarks.append(face_landmarks.landmark[idx])

        # Draw the cheeks and forehead
        for landmark in selected_landmarks:
          cx, cy = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
          cv2.circle(image, (cx, cy), 1, (250, 200, 200), -1)
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
  
cap.release()