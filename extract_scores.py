import os
import insightface
import argparse
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--dir', help='Input directory', required=True)
parser.add_argument('--output', help='Output directory', required=True)

args = parser.parse_args()

input_path = args.dir

start = time.time()


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def extract_face_scores(database_dir):
    face_scores = []
    face_labels = []
    face_methods = []

    model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection'])
    model.prepare(ctx_id=0)

    for label in listdir_nohidden(database_dir):
        label_dir = os.path.join(database_dir, label)

        for method in listdir_nohidden(label_dir):
            method_dir = os.path.join(label_dir, method)

            for image_file in listdir_nohidden(method_dir):
                image_path = os.path.join(method_dir, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = model.get(image)
                if len(faces) > 0:
                    face_score = faces[0].det_score
                    face_scores.append([face.det_score for face in faces])
                    print("Processed {}, found {} faces, DET score: {}".format(image_path, len(faces), face_score))
                else:
                    print("No faces found in the input image {}.".format(image_path))

            face_methods.append(method)

        face_labels.append(label)

    return face_scores, face_labels, face_methods


scores, labels, methods = extract_face_scores(input_path)
print("Scores: {}".format(scores))
print("Labels: {}".format(labels))
print("Methods: {}".format(methods))

# matrix = np.array(scores)
# plt.imshow(matrix, cmap='viridis')
# plt.xlabel('Face Index')
# plt.ylabel('Image Index')
# plt.colorbar(label='Detection Score')
# plt.title('Detection Scores Matrix')
# plt.show()

end = time.time()
print("Finished extracting dataset after elapsed time ({} seconds)"
      .format((end - start)))
