### Mining 

Mining is putting positives and false positives into the training set of the model.

- `classify\_on\_images.py`: run the model on the training set and generate a mining\_set folder
	- takes a while to finish of course
	- will create a folder mining\_set in the same dir, should be the same as the results/mining\_set
- `manual_classify_mining.py`: runs thought the mining\_set folder and lets you classify them, results saved in classified\_mining folder
	- key commands:
		- "p" = previous
		- "w" = save as wart
		- "c" = save as wart with cream
		- "n" = save as negative
		- "d" = save as dubious
		- "s" = save original (for later inspection)
