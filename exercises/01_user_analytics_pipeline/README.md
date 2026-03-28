# User Analytics Data Pipeline

This pipeline processes user analytics data for ML training. It loads user data from JSON files, extracts features, and prepares datasets for model training.

## Overview
The data loader processes user information including demographics and approval status to create training datasets for machine learning models.

## Expected Output
When working correctly, the script should:
1. Load data from JSON files in the directory
2. Extract relevant features (age, income) and labels (approved status)  
3. Convert to numpy arrays for ML processing
4. Display summary statistics of the loaded data
5. Report successful data loading with sample counts

## Usage
```bash
python data_loader.py
```

## Files
- `data_loader.py` - Main data processing script
- `training_data.json` - Primary training dataset
- `validation_data.json` - Validation dataset