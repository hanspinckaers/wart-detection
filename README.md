### Dependencies

- **Python 2.7.11**
- **OpenCV 3.1 + contrib**:
	NB contrib module is needed for xfeatures2d (SIFT/SURF):
	- for OS X use: <http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>
	- for Ubuntu: <http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/>
 	- for Windows use: <http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv> and download ```opencv_python-3.1.0+contrib_opencl-cp35-cp35m-win32.whl``` or ```opencv_python-3.1.0+contrib_opencl-cp35-cp35m-win_amd64.whl``` then install via ```pip install opencv_python-3*win_amd64.whl```
- **Scikit-learn** 0.17.1 (I have it built from source 0.18.dev0, but I don't use dev features anymore, so just the latest stable version should work. However, if you're seeing sklearn related errors, try to install it from source.)
- **SciPy** 0.17.1
- **Numpy** 1.11.0
- **Joblib**
- **kmajority.c**: build with ```clang -Ofast -o kmajority kmajority.c``` or ```gcc -Ofast -o kmajority kmajority.c```
- **bh_tsne (Barnes-Hut TSNE)**: To run ```test_tsne.py``` **bh\_tsne** needs to be installed, follow instructions on: <https://github.com/lvdmaaten/bhtsne>. Make sure that the bh_tsne executable in in the root folder of the project. Use cygwin on Windows for compilation.
- **spearmint**: install <https://github.com/JasperSnoek/spearmint> and it's dependencies for Bayesian Optimization


### Pipeline:

#### Part 1

- Automatically find subregions with skin lesions ```detect_warts.py```
- Manually classify skin regions ```manual_classify.py```

#### Part 2

- Gather features of images: ```features.py```
	1. For every image: 
	2. Get features (with feature detector)
	3. Describe the features (with descriptor)
- Train a bag of words: ```train.py: train_bagofwords()```, uses kmajority ```kmajority.py``` for binary features
	1. Training a bag of words is basically kmeans clustering of the described features (with non-binary features)
	2. Create a arbritary order of cluster centers (called the **vocabulary**)
- Histogram generations: ```train.py: hist_using_vocabulary()```
	1. For every image:
	2. Get features (with feature detector) (```features.py```)
	3. Describe the features
	4. Per feature description find closest cluster center in vocabulary
	5. Create a histogram of occurences of cluster centers in image
	6. Normalize histogram
- Train classifier with histograms  ```train.py: train_model()```
	1. Manually see if classifier works with certain parameters
	2. Use Bayesian Optimization to optimize these parameters
	3. Use spearmint (expected to be installed in ./spearmint)
	4. Run experiment see bay_opt/experiment.py
- Test classifier on test set

		
### Python scripts:

- *detect\_warts.py*: a python script that searches a directory for images and runs wart_detection.py on them
	- run on folder: ```python detect_warts.py dir [dirname]``` e.g. "images" as dirname
    - run on folder with parallelism: ```python detect_warts.py dir images [number-of-processes]```
    - run on file: ```python detect_warts.py images/wart-on-skin.png```
    - quit by typing any key
    - subregions are saved in the output folder

- *manual_classify.py*: this python script will show the images in a directory one-by-one for classifying the pictures by hand:
	- run on folder: ```python classify.py [dir]```

- *test_tsne.py*: the python script responsible for:
	1. extracting features of images
	2. putting features in bag of words
	3. make BOW vocabulary
	4. reassess images using vocabulary to create histograms
	5. putting histograms in tsne algorithm
	6. undoing the overlap in the tsne
	- run: ```python test_tsne.py``` (will use the folder "classified")

### Utility scripts:

- *hierarchical\_tweaked.py*: added functionality to hierarchical clustering of sklearn to return distances.

- *wart\_detection.py*: the python script responsible for finding the regions in an image with possible warts, shouldn't be called directly only via detect_warts.py.

- *image_scatter.py*: this python script contains the function needed to resolve overlap in a scatterplot with images.
