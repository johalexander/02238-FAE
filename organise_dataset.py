import argparse
import os
import pathlib
import cv2
import time
from glob import glob
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--dir', help='Input directory', required=True)
parser.add_argument('--output', help='Output directory', required=True)
parser.add_argument('--tag', help='Tag to associate to', default="original", choices=["original", "blur", "gaussian", "median", "bilateral", "pixelate"])
parser.add_argument('--resize', help='Resize images', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

input_path = args.dir
output_path = args.output
resize = args.resize
tag = args.tag

image_file_paths = glob(input_path + "*.png")
pathlib.Path(output_path + '/organised_dataset').mkdir(exist_ok=True)
start = time.time()

count_dict = defaultdict(int)
for file in image_file_paths:
    image = cv2.imread(file)
    if resize:
        resized = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    else:
        resized = image
    basename = os.path.basename(file)
    first_five = basename[:5]
    count_dict[first_five] += 1
    pathlib.Path(output_path + '/organised_dataset/' + first_five).mkdir(exist_ok=True)
    pathlib.Path(output_path + '/organised_dataset/' + first_five + '/' + tag).mkdir(exist_ok=True)
    output_file_path = output_path + "/organised_dataset/{}/{}/{}_000{}.jpg".format(first_five, tag, first_five, count_dict[first_five])
    cv2.imwrite(output_file_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print("Put {} in output path [{}], converted to JPG".format(file, output_file_path))

end = time.time()
print("Finished organising dataset after elapsed time ({} seconds), output directory: [/organised_dataset]"
      .format((end - start)))
