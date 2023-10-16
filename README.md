# Readme

## How to run

Install dependencies by running `pip install -r requirements.txt`
This code has been tested with Python 3.9 on Windows 11

Call the following from the project root to run the code:

`python data_science_practice.py` 

If you wish to see performance metrics you can run 

`python -m cProfile -s time data_science_practice.py`

The output of this program is a parquet file called `trades.parquet` which will be saved to the directory `data`. 
The input files should also be placed in this folder.