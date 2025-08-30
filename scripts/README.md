# Data Preparation Script

This directory contains the data preparation script that extracts and consolidates data processing logic from the Jupyter notebook and TextProcessor class. The script now uses Hydra configuration to control text processing settings.

## Usage

### Basic Usage

```bash
python prepare_data.py
```

This will use the default configuration from `src/config/config.yaml` and:
- Load raw data from the configured CSV file paths
- Apply text cleaning and fraud indicator extraction based on config settings
- Add engineered features
- Save prepared datasets to the configured output directory

### Configuration-based Usage

You can override configuration settings using Hydra's command-line interface:

```bash
# Disable text cleaning
python prepare_data.py preprocessing.text.apply_cleaning=false

# Disable fraud indicators
python prepare_data.py preprocessing.text.add_fraud_indicators=false

# Change input file paths and output directory
python prepare_data.py data_preparation.train_path=/path/to/train.csv data_preparation.test_path=/path/to/test.csv data_preparation.output_dir=/path/to/output

# Combine multiple overrides
python prepare_data.py preprocessing.text.apply_cleaning=false preprocessing.text.add_fraud_indicators=true data_preparation.train_path=raw_data/train.csv
```

### Configuration Files

The script uses configuration from `src/config/`:
- **config.yaml**: Main configuration file
- **preprocessing/text.yaml**: Text processing settings including:
  - `apply_cleaning`: Enable/disable text cleaning (default: true)
  - `apply_lemmatization`: Enable/disable lemmatization (default: false)  
  - `add_fraud_indicators`: Enable/disable fraud indicator extraction (default: true)
  - `max_length`: Maximum text length (default: 1024)
  - `nltk_data_dir`: NLTK data directory path

## Output Files

The script generates the following files:

1. **train_prepared.csv** - Full training dataset with all features
2. **test_prepared.csv** - Full test dataset with all features
3. **train.csv** - Training split (80% of training data)
4. **val.csv** - Validation split (20% of training data)
5. **test.csv** - Final test dataset for predictions

## Features Added

### Text Processing
- Text cleaning and normalization
- Fraud indicator extraction based on business rules
- Text length and word count features

### Engineered Features
- Return rates and ratios
- Sales growth trends
- Activity velocities
- Price-return interactions
- Seller-level aggregations
- Anomaly scores

## Changes Made

### From Notebook
Extracted the `add_feat()` function and related data preparation code from `notebooks/data_preparation.ipynb`.

### From TextProcessor
Moved the following logic from `src/dataset/processor.py`:
- Text cleaning using TextCleaner
- Text normalization
- Fraud indicator extraction using BusinessRulesChecker

The TextProcessor class is now simplified and focuses only on basic preprocessing for inference.

## Dependencies

Make sure you have the required preprocessing modules:
- `preprocessing.text.cleaner.TextCleaner`
- `preprocessing.text.normalizer.normalize_text`
- `preprocessing.text.business_rules.BusinessRulesChecker`

## Running the Script

1. Ensure your raw data files are in the correct location
2. Run the script once before training your models
3. Use the generated prepared datasets for model training and evaluation

```bash
cd scripts
python prepare_data.py
```

The script will provide progress updates and confirm successful completion.

# Model Calibration

### calibrate_temperature.py

Calibrates a trained model using temperature scaling to improve probability calibration. This addresses overconfident predictions by optimizing a temperature parameter on the validation set.

**Usage:**
```bash
python scripts/calibrate_temperature.py --config-path=config/calibration --config-name=default
```

**Configuration:**
The script uses Hydra configuration. Edit `src/config/calibration/default.yaml` to customize parameters:

```yaml
# Model and training config
model_path: "checkpoints/experiment_best.pt"
training_config: "src/config/training/default.yaml"

# Output settings
output_dir: "calibrated_models"

# Device configuration
device: "auto"  # auto, cpu, cuda

# Temperature optimization parameters
optimization:
  max_iter: 50
  lr: 0.01

# Calibration evaluation
evaluation:
  n_bins: 10  # Number of bins for ECE calculation
```

**Override parameters:**
```bash
# Override specific parameters
python scripts/calibrate_temperature.py model_path=checkpoints/my_model.pt output_dir=my_calibrated_models

# Use different config
python scripts/calibrate_temperature.py --config-name=custom_calibration
```

**Outputs:**
- `temperature_scaled_model.pt` - Calibrated model with optimal temperature
- `reliability_diagram.png` - Before/after calibration comparison
- `calibration_metrics.txt` - ECE and NLL improvement metrics

**Example workflow:**
```bash
# 1. Train your model normally first
python src/train.py --config-path=config/training --config-name=default

# 2. Calibrate the trained model
python scripts/calibrate_temperature.py \
    model_path=checkpoints/experiment_best.pt \
    training_config=src/config/training/default.yaml

# 3. Use the calibrated model for inference
# The calibrated model will have better probability calibration
```

**What it does:**
- Loads your trained model and validation data
- Optimizes temperature parameter using LBFGS on validation set
- Evaluates calibration using Expected Calibration Error (ECE)
- Generates reliability diagrams showing calibration improvement
- Saves the temperature-scaled model for inference

**When to use:**
- Your model shows overconfident predictions
- Probability calibration is important for your use case
- You need well-calibrated uncertainty estimates

# Inference Script

This directory also contains an inference script `run_inference.py` to generate predictions on a test dataset using a trained model.

## Usage

### Basic Usage

To run inference, you need to provide the path to the test data and the trained model checkpoint. You can do this via command-line overrides.

```bash
python run_inference.py test_path=/path/to/your/test_prepared.csv +model_path="/path/to/your/model.pth"
```

### Example

Using the sample test data from `examples/sample_test.csv`:

```bash
# First, prepare the sample data
python prepare_data.py data_preparation.test_path=../examples/sample_test.csv

# Then, run inference on the prepared sample data
python run_inference.py test_path=../data/test_prepared.csv +model_path="/path/to/your/model.pth"
```

### Output

The script will generate a `submission.csv` file in the root directory with two columns:
- `id`: The item ID.
- `prediction`: The predicted fraud probability (a value between 0 and 1).
