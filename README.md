# Dynamical System Prediction from Sparse Observations Using Deep Neural Networks with Voronoi Tessellation and Physics Constraints (DSOVT)

## NOAA Sea Surface Temperature Data Processing

This script processes sea surface temperature (SST) data to generate training and testing datasets. The script reads SST data from a `.mat` file (available at: [Google Drive Link](https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu)), performs interpolation to handle missing values, generates sparse sampling based on sensor numbers and variables, and saves the processed data to specified output directories as numpy arrays.

### Usage
Execute the script via command line as follows:

```bash
python NOAA_data_generation.py <output_dir>
```
Replace <output_dir> with the path to your desired output directory.

## Dependencies
The script requires the following dependencies:
- Python 3.x
- NumPy
- Pandas
- SciPy
- h5py
- tqdm

Ensure all dependencies are installed using pip:

```bash
pip install numpy pandas scipy h5py tqdm

