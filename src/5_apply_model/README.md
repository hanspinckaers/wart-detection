### Here a saved model can be used to classify a whole image

- `analyze_all.py`: creates an HTML table with one: where there is an image but model is missing classification, 2: model says image is compliant, 3: model says image is non-compliant
- `analyze_all_hist.py`: creates histogram of scores of positive images and negative images. The script uses the final\_model. This can be changed in method classify\_img.py
- `classify.py`: classifies one image (see file what to uncomment to enable heatmap visualization). It takes two arguments :
	- -i / --image: path to the image to cl
	- -d / --id: an id for the file in which the results are saved
- `classify_on_images.py`: runs classifier on all images to create regions of interest, for mining purposes
- `nms.py`: non-maximum suppression: creates an heatmap per image and thresholds this heatmap to make a decision.

- `run_on_all_images`: runs classifier on all images (just run by python run\_on\_all\_images, this can take a while!)
	- tweak the number of processes on n\_jobs
	- results will be outputted as img_<id>.txt files
