from face_roi import mat_converter
import os

in_dir = 'scamps/scamps_videos_example/'
out_dir = 'scamps/converted/'
print(os.listdir(in_dir))

for file in os.listdir(in_dir):
    print(in_dir+file)
    if os.path.isfile(in_dir+file):
        mat_converter(str(in_dir + file), out_dir)
