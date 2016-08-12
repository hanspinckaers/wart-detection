### Dependencies
 
- **OpenCV 3.1 + contrib**:
	NB contrib module is needed for xfeatures2d (SIFT/SURF):
	- for OS X use: <http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/>
	- for Ubuntu: <http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/>
 	- for Windows use: <http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv> and download ```opencv_python-3.1.0+contrib_opencl-cp35-cp35m-win32.whl``` or ```opencv_python-3.1.0+contrib_opencl-cp35-cp35m-win_amd64.whl``` then install via ```pip install opencv_python-3*win_amd64.whl```
- **Python 2.7.11**
- **Scikit-learn** built from source 0.18.dev0: I don't remember why I needed to built from source, but if you're seeing sklearn related errors, try to install it from source.
- **SciPy** 0.17.1
- **Numpy** 1.11.0
- **bh_tsne (Barnes-Hut TSNE)**: To run ```test_tsne.py``` **bh\_tsne** needs to be installed, follow instructions on: <https://github.com/lvdmaaten/bhtsne>. Make sure that the bh_tsne executable in in the root folder of the project.

### Python scripts:

- *detect\_warts.py*: a python script that searches a directory for images and runs wart_detection.py on them
	- run on folder: ```python detect_warts.py dir [dirname]``` e.g. "images" as dirname
    - run on folder with parallelism: ```python detect_warts.py dir images [number-of-processes]```
    - run on file: ```python detect_warts.py images/wart-on-skin.png```
    - quit by typing any key
    - subregions are saved in the output folder

- *classify.py*: this python script will show the images in a directory one-by-one for classifying the pictures by hand:
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


### Older notes


- brew reinstall libpng --universal
- brew reinstall freetype --universal
- PYTHON_CONFIGURE_OPTS="--enable-shared --enable-unicode=ucs2"  pyenv install 2.7.11 
- install virtualenv
- copy cv2.so
- pip install numpy
- pip install matplotlib 

```
// should global python be 2.7.11?
brew install opencv3
// This puts 2 files namely, cv.py and cv2.so in /usr/local/lib/python2.7/site-packages
// In order to use this from the virtualenv, copy these two files and place them in your virtualenv's site-packages
cp /usr/local/lib/python2.7/site-packages/cv* ~/opencv3/lib/python2.7/site-packages/

Problem Cause In mac os image rendering back end of matplotlib (what-is-a-backend to render using the API of Cocoa by default). There is Qt4Agg and GTKAgg and as a back-end is not the default. Set the back end of macosx that is differ compare with other windows or linux os.

I resolve this issue following ways:

I assume you have installed the pip matplotlib, there is a directory in you root called ~/.matplotlib.
Create a file ~/.matplotlib/matplotlibrc there and add the following code: "backend: TkAgg"
From this link you can try different diagram.

### Problemen met MacVim en YouCompleteMe and pyenv:

uninstall pyenv
remove youcompleteme
remove macvim
brew install macvim
install youcompleteme
brew install pyenv
(weet niet of enable-shared nodig is)
PYTHON_CONFIGURE_OPTS="--enable-shared --enable-unicode=ucs2"  pyenv install 2.7.11 
```
