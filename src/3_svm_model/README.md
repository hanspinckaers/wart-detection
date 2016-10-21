### Train a model

Run with: ```python train.py 0.9 1.4```

- first argument is gamma parameter of SVM (10^0.9)
- second argument is kernel parameter of SVM (10^1.4)
		
- ```features.py```: gather features of images (location to images is hardcoded at the end of file and in cross_validate_with_participants): 
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
- Do k-fold validation

- See documentation of cross_validate_with_participants in code for more.

### Utility scripts:

- *features.py*: this script has function for getting features out of images

- *divide.py*: this script divides the data into folds for the k-fold validation. 
