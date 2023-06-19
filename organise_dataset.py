import argparse
import os
import pathlib
import cv2
import time
from glob import glob
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--dir', help='Input directory', required=True)

args = parser.parse_args()

input_path = args.dir

image_file_paths = glob(input_path + "*.png")
pathlib.Path(input_path + '/organised_dataset').mkdir(exist_ok=True)
start = time.time()

count_dict = defaultdict(int)
for file in image_file_paths:
    image = cv2.imread(file)
    resized = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    basename = os.path.basename(file)
    first_five = basename[:5]
    count_dict[first_five] += 1
    pathlib.Path(input_path + '/organised_dataset/' + first_five).mkdir(exist_ok=True)
    output_file_path = file.replace(basename, "organised_dataset/{}/{}_000{}.jpg".format(first_five, first_five, count_dict[first_five]))
    cv2.imwrite(output_file_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print("Put {} in output path [{}], converted to JPG".format(file, output_file_path))

end = time.time()
print("Finished organising dataset after elapsed time ({} seconds), output directory: [/organised_dataset]"
      .format((end - start)))
