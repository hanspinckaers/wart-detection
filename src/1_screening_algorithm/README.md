### Pipeline:

#### Part 1

- Automatically find subregions with skin lesions ```detect_warts.py```
- Manually classify skin regions ```manual_classify.py```

### Python scripts:

- *detect\_warts.py*: a python script that searches a directory for images and runs wart_detection.py on them
	- run on folder: ```python detect_warts.py dir [dirname]``` e.g. "images" as dirname
    - run on folder with parallelism: ```python detect_warts.py dir ../../images/test_images [number-of-processes]```
    - run on file: ```python detect_warts.py ../../images/test_images/wart-on-skin.png```
    - quit by typing any key
    - subregions are saved in a output folder

- *manual\_classify.py*: this python script will show the images in a directory one-by-one for classifying the pictures by hand:
	- run on folder: ```python classify.py [dir]```
	- e.g.: ```python manual_classify.py ../../results/naive_algorithm_per_img```
	- results are saved in new folder classified
	- key commands:
		- "p" = previous
		- "w" = save as wart
		- "c" = save as wart with cream
		- "n" = save as negative
		- "d" = save as dubious
		- "s" = save original (for later inspection)

### Utility scripts:

- *hierarchical\_tweaked.py*: added functionality to hierarchical clustering of sklearn to return distances.

- *find\_wart.py*: the python script responsible for finding the regions in an image with possible warts, shouldn't be called directly, used by detect_warts.py.

- *utilities.py*: methods to segment an image with hierarchical clustering, search the skin cluster, 
