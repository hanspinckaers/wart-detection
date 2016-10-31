### Bayesian optimization

Spearmint needs to be installed for this to work: https://github.com/JasperSnoek/spearmint or see install instructions in root folder README.

Run with: 
```
~/spearmint/spearmint/bin/spearmint ./config.pb --drive=local --method=GPEIOptChooser --max-concurrent=2
```
If it doesn't work check the path to the spearmint binary. Max-concurrent is the number of processes, related to number of cores.

config.pb defines the boundaries/ranges of the parameters to test
