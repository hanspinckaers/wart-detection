OpenCV 3.1
Python 2.7.11

### Installation on Mac:


brew reinstall libpng --universal
brew reinstall freetype --universal
PYTHON_CONFIGURE_OPTS="--enable-shared --enable-unicode=ucs2"  pyenv install 2.7.11 
install virtualenv
copy cv2.so
pip install numpy
pip install matplotlib 

// should global python be 2.7.11?
brew install opencv3
// This puts 2 files namely, cv.py and cv2.so in /usr/local/lib/python2.7/site-packages
// In order to use this from the virtualenv, copy these two files and place them in your virtualenv's site-packages
cp /usr/local/lib/python2.7/site-packages/cv* ~/opencv3/lib/python2.7/site-packages/

### Problem Cause In mac os image rendering back end of matplotlib (what-is-a-backend to render using the API of Cocoa by default). There is Qt4Agg and GTKAgg and as a back-end is not the default. Set the back end of macosx that is differ compare with other windows or linux os.

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
