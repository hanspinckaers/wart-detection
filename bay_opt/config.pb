language: PYTHON
name:     "experiment" 

variable {
	name: "svm_C"
	type: FLOAT
	size: 1
	min:  -4
	max:  3
}

variable {
	name: "svm_gamma"
	type: FLOAT
	size: 1
	min:  -4
	max:  3
}

variable {
	name: "bow_size"
	type: INT
	size: 1
	min:  100
	max:  2000
}

variable {
	name: "nfeatures"
	type: INT
	size: 1
	min:  10
	max:  100
}

variable {
	name: "contrastThreshold"
	type: FLOAT
	size: 1
	min:  0.0001
	max:  0.1
}

variable {
	name: "edgeThreshold"
	type: FLOAT
	size: 1
	min:  3
	max:  100
}

variable {
	name: "sigma"
	type: FLOAT
	size: 1
	min:  0.2
	max:  0.9 
}
