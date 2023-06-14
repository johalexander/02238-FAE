import cv2
import argparse
import os
import pathlib
import time
from glob import glob
from anonymise import get_anonymisation_method, anonymise_faces

parser = argparse.ArgumentParser()

parser.add_argument('--dir', help='Input directory', required=True)
parser.add_argument('--method', help='Image anonymisation method',
                    choices=["blur", "gaussian", "median", "bilateral", "pixelate"],
                    default="blur")

args = parser.parse_args()

input_path = args.dir
input_method = args.method
method = get_anonymisation_method(args.method)

image_file_paths = glob(input_path + "*.png")
pathlib.Path(input_path + '/' + input_method).mkdir(exist_ok=True)
start = time.time()

for file in image_file_paths:
    image = cv2.imread(file)
    anonymise_faces(image, method)
    basename = os.path.basename(file)
    output_file_path = file.replace(basename, "{}/{}".format(input_method, basename, input_method))
    cv2.imwrite(output_file_path, image)
    print("Anonymised {} with method [{}], output: {}".format(file, input_method, output_file_path))

end = time.time()
print("Finished anonymising images after elapsed time ({} seconds), output directory: [/{}]"
      .format((end - start), input_method))
