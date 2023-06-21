import os
import insightface
import argparse
import cv2
import numpy as np
import time
import pathlib
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()

parser.add_argument('--dir', help='Input directory', required=True)
parser.add_argument('--output', help='Output directory', required=True)
parser.add_argument('--save', help='Save output', action=argparse.BooleanOptionalAction)
parser.add_argument('--plot', help='Should plot', action=argparse.BooleanOptionalAction)
parser.add_argument('--tags', help='Tag to associate to', default="original")
# "original,blur,gaussian,median,bilateral,pixelate"

args = parser.parse_args()

input_path = args.dir
output_path = args.output
save_output = args.save
plot = args.plot
targets = args.tags.split(',')

start = time.time()
shutil.rmtree(output_path + '/output', ignore_errors=True)
pathlib.Path(output_path + '/output').mkdir(exist_ok=True)


# Hides dirs and sorts
def skip_hidden_and_sort_dir(path):
    nohiddendir = os.listdir(path)
    nohiddendir.sort()
    for f in nohiddendir:
        if not f.startswith('.'):
            yield f


def extract_face_scores(database_dir):
    face_scores = []
    face_labels = []
    face_tag = []

    model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'], rcond=None)
    model.prepare(ctx_id=0, det_size=(128, 128))

    for label in skip_hidden_and_sort_dir(database_dir):
        label_dir = os.path.join(database_dir, label)

        for method in skip_hidden_and_sort_dir(label_dir):
            if method not in targets:
                continue

            method_dir = os.path.join(label_dir, method)

            for image_file in skip_hidden_and_sort_dir(method_dir):
                image_path = os.path.join(method_dir, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = model.get(image)
                if len(faces) > 0:
                    face_scores.append(faces[0].embedding)
                    face_tag.append(method)
                    face_labels.append(label)
                    print("Processed {}, found {} faces, DET score: {}".format(image_path,
                                                                               len(faces), faces[0].det_score))
                else:
                    print("No faces found in the input image {}.".format(image_path))

    return face_scores, face_labels, face_tag


scores, labels, tags = extract_face_scores(input_path)
#print("Scores: {}".format(scores))
#print("Labels: {}".format(labels))
#print("Tags: {}".format(tags))

if save_output:
    print("Saving output in [/output]")

    embeddings = np.stack(scores)
    similarity_matrix = cosine_similarity(scores)
    similarity_scores = similarity_matrix[np.triu_indices(len(embeddings), k=1)]
    np.save(output_path + '/output/' + 'embeddings.npy', similarity_scores)

    # Saving these files takes a lot of memory management
    # Batching the saving seems to alleviate some of the memory hangs

    batch_size = 1000
    pairs = []

    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if labels[i] == labels[j]:
                pairs.append((scores[i], scores[j]))

    arr_pairs = np.array(pairs)

    num_batches = len(arr_pairs) // batch_size + 1
    similarity_scores = []

    for batch in range(num_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(arr_pairs))

        batch_pairs = arr_pairs[start:end]
        batch_similarity_matrix = cosine_similarity(batch_pairs[:, 0], batch_pairs[:, 1])
        batch_similarity_scores = batch_similarity_matrix[np.triu_indices(len(batch_pairs), k=1)]
        similarity_scores.extend(batch_similarity_scores)

    similarity_scores = np.array(similarity_scores)
    np.save(output_path + '/output/' + 'mated.npy', similarity_scores)

    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if labels[i] != labels[j]:
                pairs.append((scores[i], scores[j]))

    arr_pairs = np.array(pairs)

    num_batches = len(arr_pairs) // batch_size + 1
    similarity_scores = []

    for batch in range(num_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(arr_pairs))

        batch_pairs = arr_pairs[start:end]
        batch_similarity_matrix = cosine_similarity(batch_pairs[:, 0], batch_pairs[:, 1])
        batch_similarity_scores = batch_similarity_matrix[np.triu_indices(len(batch_pairs), k=1)]
        similarity_scores.extend(batch_similarity_scores)

    similarity_scores = np.array(similarity_scores)
    np.save(output_path + '/output/' + 'non_mated.npy', similarity_scores)


if plot:
    similarity_matrix = cosine_similarity(scores)
    fig, ax = plt.subplots()

    im = ax.imshow(similarity_matrix, cmap='hot')
    norm = mcolors.Normalize(vmin=similarity_matrix.min(), vmax=similarity_matrix.max())

    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            text_color = 'white' if norm(similarity_matrix[i, j]) < 0.5 else 'black'
            text = ax.text(j, i, '{:.2f}'.format((similarity_matrix[i, j])),
                           ha='center', va='center', color=text_color, fontsize=4)

    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_xlabel('Image Index')
    ax.set_ylabel('Image Index')
    ax.set_title('Similarity Score Matrix')

    unique_tag = np.unique(tags)
    tag_groups = {tag: [] for tag in tags}

    for i, tag in enumerate(tags):
        tag_groups[tag].append(i)

    color_map = {'original': 'blue', 'blur': 'red', 'median': 'magenta', 'bilateral': 'pink', 'gaussian': 'purple', 'pixelate': 'green'}

    for t, indices in tag_groups.items():
        if indices:
            color_tags = [tags[i] for i in indices]

            for idx, tag in zip(indices, color_tags):
                rect = plt.Rectangle((idx - 0.5, idx - 0.5),
                                     1, 1,
                                     fill=False, edgecolor=color_map[tag], linewidth=2)
                ax.add_patch(rect)

    plt.show()

end = time.time()
print("Finished extracting scores after elapsed time ({} seconds)"
      .format((end - start)))
