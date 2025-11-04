# Project Overview
AttentionMOI: A four-class classifier for the GBM subtype

## Prerequisites
- Python 3.8+
- Recommended: virtual environment (Pytorch conda)

## Directory Structure
├── best_model/              # Best-performing model + best logs
├── data/                    # Raw input data
├── logs/                    # Training logs and experiment records
├── models/                  # Checkpoints saved during training
├── preprocessed/            # Data generated after preprocessing
├── results/                 # Visualization and evaluation outputs
├── data_preprocessing.py    # Process raw data -> preprocessed/
├── train_sam.py             # Train model using preprocessed data
├── test.py                  # Load best model and generate visualization results
└── README.md

## Usage Workflow (Recommended Order)
Place raw files into data/
Run data preprocessing to generate preprocessed/
Train the model using files in preprocessed/
Test the model and generate visualization outputs into results/

## Data Preprocessing

-data_preprocessing.py processes all files in the data/ directory, performing cleaning, formatting, and structuring to prepare data for model training and testing.

-Input Directory data/

-Output Directory preprocessed/

-Run Preprocessing
``` python data_preprocessing.py```

### Notes 
-Both training and testing  use data from the preprocessed/ directory.
-Ensure preprocessing completes successfully before moving to the training step.

## Model Training

-train_sam.py trains the model using preprocessed data and automatically saves checkpoints and the best-performing model.

-Input Directory preprocessed/

-Output Directories
--models/        # Regular training checkpoints
--best_model/    # Best-performing model + best logs
--logs/          # Additional training logs

-Run Training
```python train_sam.py```

## Testing and Visualization
-test.py loads the best model from the best_model/ directory and evaluates it on the test set.
-Visualization outputs such as figures, prediction plots, and evaluation metrics are saved in the results/ directory.

-Output Directory results/

-Run Testing
```python test.py```

### Notes
-test.py automatically uses the model stored in best_model/.
-Visualization outputs (figures, curves, etc.) will appear in the results/ folder.