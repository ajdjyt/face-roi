import cv2
import mediapipe as mp
import numpy as np
import h5py
import imageio as iio

def get_roi_values()->tuple:
    forehead_outline = [21, 71, 70, 46, 225, 224, 223, 222, 221, 193, 168, 417, 441, 442, 443, 444, 445, 276, 300, 301, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54]
    forehead_inners = [68, 104, 69, 108, 151, 337, 299, 333, 298] + [43, 105, 66, 107, 9, 336, 296, 334, 293] + [53, 52, 65, 55, 8, 285, 295, 282, 283]

    left_cheek_outline = [234, 227, 116, 117, 118, 101, 36, 203, 165, 92, 186, 57, 43 ] + [202, 210, 169] + [150, 136, 172, 58, 132, 93]
    left_cheek_inners = [137, 123, 50, 205, 206] + [177, 147, 187, 207, 216] + [215, 213, 192, 214, 212] + [138, 135]

    right_cheek_outline = [454, 447, 345, 346, 347, 330, 266, 423, 391, 322, 410, 287, 273 ] + [422, 430, 394] + [379, 365, 397, 288, 361, 323]
    right_cheek_inners = [366, 352, 280, 425, 426] + [401, 376, 411, 427, 436] + [435, 433, 416, 434, 432] + [367, 364]

    outlines = [forehead_outline, left_cheek_outline, right_cheek_outline]
    inners = [forehead_inners + left_cheek_inners + right_cheek_inners]
    landmarks=[i for sub in outlines+inners for i in sub]
    return outlines, inners, landmarks

def get_face_roi(stream:cv2.VideoCapture, det_conf:float=0.5, track_conf:float=0.5):
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf) as face_mesh:

      outlines, inners, landmarks = get_roi_values()
         
      while stream.isOpened():
        success, image = stream.read()
        if not success:
          return

        # Process Image with face_mesh
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display results
        if results.multi_face_landmarks:
            original = image.copy()
            mask = np.zeros_like(image)

            for face_landmarks in results.multi_face_landmarks:

                selected_landmarks = [face_landmarks.landmark[idx] for idx in landmarks]

                # Plot the outline on mask for every ROI
                for selected_outline in outlines:
                  selected_points = np.array([(int(face_landmarks.landmark[landmark].x * image.shape[1]), int(face_landmarks.landmark[landmark].y * image.shape[0])) for landmark in selected_outline])
                  cv2.fillPoly(mask, [selected_points], (255, 255, 255))

                # Plot the mesh landmarks in every ROI
                for landmark in selected_landmarks:
                  cx, cy = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                  cv2.circle(image, (cx, cy), 1, (250, 200, 200), -1)

            # Apply the mask and mesh to the image
            meshed_image = image
            meshed_and_masked_image = cv2.bitwise_and(meshed_image, mask)
            masked_image = cv2.bitwise_and(original, mask)
            yield masked_image, meshed_image, meshed_and_masked_image
        else:
            print("No face detected")
            continue
    stream.release()

def display_roi(stream:cv2.VideoCapture):

    roi = get_face_roi(stream)
    try:
        while True:
            masked_image, meshed_image, meshed_and_masked_image = next(roi)
            image = cv2.hconcat([masked_image, meshed_image, meshed_and_masked_image])
            cv2.imshow('Masked Face, Meshed Face, Masked and Meshed Face', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    except StopIteration:
        print("End of Stream")
        return

def mat_converter(path:str, out_path:str='', fps:float=30.0)->str:
    import os
    file_name = os.path.basename(path).strip()
    if out_path == '':
        out_path = os.path.join(os.path.dirname(path), 'converted')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print(f"output path set to {out_path}")
    mat = h5py.File(path, 'r')
    frames = np.array(mat['RawFrames'])
    out_name = os.path.join(out_path, file_name + ".avi")
    writer = iio.get_writer(out_name, fps=fps)
    for frame_idx in range(frames.shape[-1]):
        frame = frames[:, :, :, frame_idx]*255
        frame = np.transpose(frame, (2, 1, 0)).astype(np.uint8)
        writer.append_data(frame)
    writer.close()
    return out_name

def mat_loader(path:str):
    mat = h5py.File(path, 'r')
    frames = mat['RawFrames']
    for frame in frames:
        frame = frame.astype(np.uint8)
    raise NotImplemented

def get_roi_video(path:str, source_type:str="video", extraction_type:str='masked')-> cv2.VideoCapture:
    
    # Check extraction type
    if extraction_type not in ['masked', 'meshed', 'meshed_and_masked']:
        raise TypeError("Supported extraction_types are masked,meshed,meshed_and_masked")
   
   # Case source_type = video
    vid_types = [".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".flv"]
    if source_type=="video":
        
        # get file names and get directory names, also check file type is supported
        import os
        file_name = os.path.basename(path).strip()
        if not os.path.splitext(file_name)[-1] in vid_types:
            raise TypeError("File Type Not supported")
        in_dir = os.path.dirname(path)
        out_dir = in_dir + 'roi'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        name = out_dir + file_name
        
        # Open writer using imageio
        import imageio as iio
        reader = iio.get_reader(path)
        writer = iio.get_writer(name, fps=reader.get_meta_data()['fps'])
        reader.close()
        
        roi = get_face_roi(cv2.VideoCapture(path))
        
        # Write data while frames exist
        try:
            while True:
                masked_image, meshed_image, meshed_and_masked_image = next(roi)
                if extraction_type == 'masked':
                    data = masked_image
                elif extraction_type == 'meshed':
                    data = meshed_image
                elif extraction_type == 'meshed_and_masked':
                    data = meshed_and_masked_image
                writer.append_data(data)
        
        # When generation stops
        except StopIteration:
            None
        
        finally:
            writer.close()
            print(f"Created: {out_dir}")
        
        # Return video cap of roi video
        return cv2.VideoCapture(name)

    elif source_type=="mat":
        # Convert mat file to video and call self with "video"
        converted = mat_converter(path)
        return get_roi_video(converted, "video", extraction_type=extraction_type)
    
    else:
        raise TypeError("Non compatible Source Type Given")