### This code is for the human versus model experiment

- `pick\_images`: can be used to randomly pick images fomr our test set
- `server.py`: runs a simple server to serve the quiz (just run with `python server.py`)
- `calculate.py` does simple statistics on the results (kappa etc)

The results folder shows a csv file per participant, schema:
`participants id, question number, x coordinate of wart, y coordinate of wart, classified as`

(can be confusing because wart = negative and cream = positive)
