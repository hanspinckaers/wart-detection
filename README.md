### Dependencies

- **Python 2.7.11**
- **OpenCV 3.1 + contrib**:
	NB contrib module is needed for xfeatures2d (SIFT/SURF):
	- for OS X use: <http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>
	- for Ubuntu: <http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/>
 	- for Windows use: <http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv> and download ```opencv_python-3.1.0+contrib_opencl-cp35-cp35m-win32.whl``` or ```opencv_python-3.1.0+contrib_opencl-cp35-cp35m-win_amd64.whl``` then install via ```pip install opencv_python-3*win_amd64.whl```
	- Ubuntu
		1. `sudo apt-get install build-essential cmake git pkg-config`
		2. `sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev`
		3. `sudo apt-get install libgtk2.0-dev`
		4. `sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev`
		5. `sudo apt-get install libatlas-base-dev gfortran`
		6. install pip (could also be there already): `wget https://bootstrap.pypa.io/get-pip.py`
		7. `sudo python get-pip.py`
		8. install python (could be there already) (could be there already) (could be there already) (could be there already) (could be there already) (could be there already) (could be there already) (could be there already) (could be there already) `sudo apt-get install python2.7-dev`
		9. pip install numpy
		10. `cd ~`
		11. `git clone https://github.com/Itseez/opencv.git`
		12. `cd opencv`
		13. `git checkout 3.1.0`
		14. `cd ~`
		15. `git clone https://github.com/Itseez/opencv\_contrib.git`
		16. `cd opencv\_contrib`
		17. `git checkout 3.1.0`
		18. `cd ~/opencv`
		19. `mkdir build`
		20. `cd build`
		21. ```cmake -D CMAKE_BUILD_TYPE=RELEASE \
						-D CMAKE_INSTALL_PREFIX=/usr/local \
						-D INSTALL_C_EXAMPLES=ON \
						-D INSTALL_PYTHON_EXAMPLES=ON \
						-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
						-D BUILD_EXAMPLES=OFF ..```
		22. `make -j1`
		23. `sudo make install`
		24. `sudo ldconfig`
		25. now try with: `python` -> `import cv2`

####Install dependencies:
```pip install numpy, scipy, scikit-learn, joblib```

####Python dependencies:
- **Scikit-learn** 0.17.1 (I have it built from source 0.18.dev0, but I don't use dev features anymore, so just the latest stable version should work. However, if you're seeing sklearn related errors, try to install it from source.)
- **SciPy** 0.17.1
- **Numpy** 1.11.0
- **Joblib**

####Build from source:
- **kmajority.c**: 
	1. `cd src/2\_compare\_detectors/`
	2. ```clang -Ofast -o kmajority kmajority.c``` or ```gcc -Ofast -o kmajority kmajority.c```
- **bh_tsne (Barnes-Hut TSNE)** this should work: 
	1. `cd src/2\_compare\_detectors/`
	2. `git clone https://github.com/lvdmaaten/bhtsne.git`
	3. `cd bhtsne`
	4. `git checkout 843c0909f293bb4b4c1584e7030b3623b5cec224`
	5. `g++ sptree.cpp tsne.cpp -o bh\_tsne -O2`
	6. copy bh_tsne binary to 2\_compare\_detectors
	7. copy bhtsne.py to 2\_compare\_detectors
- **spearmint**: install <https://github.com/JasperSnoek/spearmint> and it's dependencies for Bayesian Optimization
	1. `cd ~`
	2. `apt-get install python-protobuf`
	3. `git clone https://github.com/JasperSnoek/spearmint.git`
	4. `cd spearmint/spearmint`
	5. `bin/make_protobufs`

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
