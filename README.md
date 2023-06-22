# 02238-FAE
This repository is an implementation for Face Anonymisation Experiments (FAE) in a course at DTU: 02238 Biometric Systems.
Course site: https://christoph-busch.de/teaching-biometric-systems.html

This project has a few different focuses:
* Detecting and anonymising faces in a dataset
  * Tested on FERET and FRGC image sets
* Organising datasets for probing
* Running [InsightFace](https://github.com/deepinsight/insightface)'s RetinaFace-10GF ([buffalo-l](https://github.com/deepinsight/insightface/tree/master/model_zoo)) detection model on anonymised dataset
* Extracting similarity scores for the dataset
* A Jupyter Notebook for visualisations of data

## Setup

Setup a Python virtual environment. Tested on Python 3.10.

To install requirements for the project, please run 
```
pip install -r requirements.txt
``` 
from `root`, and `~/DET/` if running the Jupyter Notebook.

## Preparing dataset
This section details how to generate datasets as provided under `Prepared dataset`.

1. Download an image dataset e.g. FERET or FRGC
2. Anonymise the images in the dataset with the preferred method (`"blur", "gaussian", "median", "bilateral", "pixelate"`)
    ```
    python anonymise_dataset.py --dir "/FERET/reference/" --method median
    ```
   This will generate a new folder with the `method` name in the directory. **This step should be repeated for each method.**
3. To get a uniform probing dataset, we need to organise the images generated. Run
   ```
   python organise_dataset.py 
   --dir "/FERET/reference/median/" 
   --output "/FERET/reference/" 
   --tag median
   ```
   **Repeat this for each method**, e.g.:
   ```
   python organise_dataset.py 
   --dir "/FERET/reference/blur/" 
   --output "/FERET/reference/" 
   --tag blur
   --resize
   ```
   & `original`
   ```
   python organise_dataset.py 
   --dir "/FERET/reference/" 
   --output "/FERET/reference/"
   --resize
   ```
   The collected output dataset will be located in the output directory under `/organised_dataset/`. This will per default resize the image to (112, 112), but this can be disabled by toggling `--resize`/`--no-resize`.

### Prepared dataset
For convenience, prepared datasets (original + anonymised images and mated vs non-mated scores) [are provided here](https://github.com/johalexander/02238-FAE/tree/main/preformatted_datasets).

A Juypter implementation for performance on the provided dataset is [available here](https://github.com/johalexander/02238-FAE/blob/main/DET/FAE_Performance.ipynb).

## Running face detection and extracting scores
To extract scores from the dataset, execute the following on the organised dataset (with `--save`)
```
python extract_scores.py --dir /organised_dataset/ --output /output/ --save --no-plot
```
To extract plot data, run it with `--plot` and `--no-save`, e.g.
```
python extract_scores.py --dir /organised_dataset/ --output /output/ --no-save --plot
```
