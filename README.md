## Installation

```git clone https://github.com/HansPinckaers/wart-detection.git```

### Dependencies

- **Python 2.7.11**
- **OpenCV 3.1 + contrib**:
	NB contrib module is needed for xfeatures2d (SIFT/SURF):

	- Ubuntu:
```
1. sudo apt-get install build-essential cmake git pkg-config
2. sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev (could be libtiff5-dev)
3. sudo apt-get install libgtk2.0-dev
4. sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
5. sudo apt-get install libatlas-base-dev gfortran (gfortran not needed if install fails)

# install pip (could also be there already): 
6. wget https://bootstrap.pypa.io/get-pip.py
7. sudo python get-pip.py

# install python (could be there already)
8. sudo apt-get install python2.7-dev
9. sudo pip install numpy
10. cd ~

# install opencv3
11. git clone https://github.com/Itseez/opencv.git
12. cd opencv
13. git checkout 3.1.0
14. cd ~
15. git clone https://github.com/Itseez/opencv\_contrib.git
16. cd opencv\_contrib
17. git checkout 3.1.0
18. cd ~/opencv
19. mkdir build
20. cd build
21. cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D INSTALL_C_EXAMPLES=OFF \
		-D INSTALL_PYTHON_EXAMPLES=ON \
		-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
		-D BUILD_EXAMPLES=OFF ..
For Ubuntu 17 add -D ENABLE_PRECOMPILED_HEADERS=OFF
22. make -j1
23. sudo make install
24. sudo ldconfig

# now try with: 
25. python
26. python>> import cv2

It works when there are no errors.
```
- **OpenCV 3.1 + contrib** original install guides (ignore if above worked):

	- for OS X use: <http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>
	- for Ubuntu: <http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/>
	- for Windows use: <http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv> and download ```opencv_python-3.1.0+contrib_opencl-cp35-cp35m-win32.whl``` or ```opencv_python-3.1.0+contrib_opencl-cp35-cp35m-win_amd64.whl``` then install via ```pip install opencv_python-3*win_amd64.whl```

### Install dependencies:
```sudo pip install numpy, scipy, scikit-learn, joblib```

### Python dependencies:
- **Scikit-learn** 0.17.1 (I have it built from source 0.18.dev0, but I don't use dev features anymore, so just the latest stable version should work. However, if you're seeing sklearn related errors, try to install it from source.)
- **SciPy** 0.17.1
- **Numpy** 1.11.0
- **Joblib**

### Build from source:
- **kmajority.c**: 
```
cd ~/wart-detection/src/2_compare_detectors/
clang -Ofast -o kmajority kmajority.c
or:
gcc -Ofast -o kmajority kmajority.c
```

- **bh_tsne (Barnes-Hut TSNE)** this should work: 
```
cd ~/wart-detection/src/2_compare_detectors
git clone https://github.com/lvdmaaten/bhtsne.git`
cd bhtsne
git checkout 843c0909f293bb4b4c1584e7030b3623b5cec224
g++ sptree.cpp tsne.cpp -o bh\_tsne -O2
```
- copy bh_tsne binary to 2\_compare\_detectors
- copy bhtsne.py to 2\_compare\_detectors
	
- **spearmint**: install <https://github.com/JasperSnoek/spearmint> and it's dependencies for Bayesian Optimization with:
```
cd ~
sudo pip install protobuf
git clone https://github.com/JasperSnoek/spearmint.git
cd spearmint/spearmint
bin/make_protobufs
```
## Test code (should run when everything is setup right):
```
# run naive algorithm on image
cd src/1_screening_algorithm/ 
python detect_warts.py ../../images/test_images/wart-on-skin.png
# run model on img
cd src/5_apply_model/ 
python classify.py -i ../../images/test_images/wart-on-skin.png -d temp
```

## Pipeline
1. src/0_exploring: This code makes a histogram of the data (n warts per participants etc)
2. src/1_screening_algorithm: this is the *naive* algorithm to find skin lesion / warts
3. src/2_compare_detectors: comparing detectors and descriptors with tsne
4. src/3_svm_model: training the svm model
5. src/4_bayesian_optimization: bayesian optimization of svm model with spearmint
6. src/5_apply_model: applying the model to all the training images (or later test images)
7. src/6_mining: classifying the false positive regions in the training images
8. src/7_humanvsmodel: human versus model experiment

## Overall:
1. Gather features (feature detector)
2. Describe features (feature descriptor)
3. Train bag of words
4. Make histograms per image
5. Train model with histograms and class
6. Test model with histogram from test set
