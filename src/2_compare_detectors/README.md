		
### Python scripts:

- *test\_detectors*: this starts a script that will make tsne for all the different detectors and descriptors mentioned in the file

- *tsne\_scatter.py*: this script has a few hardcoded references to folders at the end of the file, it is easier to use this script by using test\_detectors.py
	- 1st arg: detector\_name
    - 2nd arg: descriptor\_name
    - 3rd arg: n\_features
    - 4th arg: sensitivity (0, 1 or 2)
    - 5th arg: bow\_size
	- e.g. ```python tsne_scatter.py SIFT SIFT 10 2 500```

### Utility scripts:

- *bh\_tsne.py*: Barnes-Hut TNSE implementation for python

- *tsne.py*: tsne implementation in python, not getting used if I remember correctly
