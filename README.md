# Naive Bayes Classification System

A Python implementation of a Naive Bayes classifier with k-fold cross-validation, confusion matrix analysis, model persistence, and comprehensive performance metrics.

## Features

- **From-scratch implementation** of Naive Bayes algorithm with Laplace smoothing
- **Command-line interface** for automated workflows and scripting
- **Model persistence** - Train once, predict many times with saved models
- **K-fold cross-validation** for robust model evaluation
- **Confusion matrix generation** with detailed performance metrics
- **Comprehensive metrics**: Precision, Recall, Specificity, and F1-score
- **Flexible data format** supporting custom categorical attributes via meta files

## Getting Started

### Prerequisites

- Python 3.x
- No external libraries required (uses only Python standard library)

### Installation
```bash
git clone 
cd 
```

### Quick Start
```bash
# K-fold cross-validation
python3 main.py --mode kfold --data voting.data --meta voting.meta --k 5

# Train and save a model
python3 main.py --mode train --data voting.train --meta voting.meta --model my_model.pkl

# Make predictions with saved model
python3 main.py --mode predict --input voting.test --output predictions.csv --model my_model.pkl

# Train and evaluate in one step
python3 main.py --mode evaluate --train voting.train --test voting.test --meta voting.meta
```

## Usage Modes

The classifier supports four operation modes:

### 1. K-Fold Cross-Validation (`kfold`)
Evaluate model performance using k-fold cross-validation.
```bash
python3 main.py --mode kfold --data voting.data --meta voting.meta --k 10
```

**Arguments:**
- `--data`: Path to dataset file
- `--meta`: Path to meta file describing attributes
- `--k`: Number of folds (default: 5)

**Output:**
- Accuracy score for each fold
- Precision, Recall, Specificity, and F1-score per class for each fold
- Average accuracy across all folds

### 2. Train (`train`)
Train a model and save it for later use.
```bash
python3 main.py --mode train --data voting.train --meta voting.meta --model voting_classifier.pkl
```

**Arguments:**
- `--data`: Path to training data file
- `--meta`: Path to meta file
- `--model`: Path to save the trained model (default: `nbc_model.pkl`)

**Output:**
- Saved model file (.pkl)
- Training statistics (classes, attributes, instance count)

### 3. Predict (`predict`)
Load a saved model and classify new data.
```bash
python3 main.py --mode predict --input new_data.csv --output predictions.csv --model voting_classifier.pkl
```

**Arguments:**
- `--input`: Path to file with data to classify
- `--output`: Path to save predictions
- `--model`: Path to saved model file (default: `nbc_model.pkl`)

**Output:**
- Output file with predictions appended to each instance

### 4. Evaluate (`evaluate`)
Train on training set and evaluate on test set with detailed metrics.
```bash
python3 main.py --mode evaluate --train voting.train --test voting.test --meta voting.meta
```

**Arguments:**
- `--train`: Path to training data file
- `--test`: Path to test data file
- `--meta`: Path to meta file

**Output:**
- Accuracy and error rate
- Confusion matrix
- Precision, Recall, Specificity, and F1-score per class

## Data Format

### .meta File Format
Defines attributes and their possible values, plus class labels:
```
attribute1:value1,value2,value3
attribute2:valueA,valueB
class:label1,label2,label3
```

**Example (voting.meta):**
```
handicapped-infants:y,n,unknown
water-project-cost-sharing:y,n,unknown
adoption-of-budget-resolution:y,n,unknown
physician-fee-freeze:y,n,unknown
el-salvador-aid:y,n,unknown
religious-groups-in-schools:y,n,unknown
anti-satellite-test-ban:y,n,unknown
aid-to-nicaraguan-contras:y,n,unknown
mx-missile:y,n,unknown
immigration:y,n,unknown
synfuels-corporation-cutback:y,n,unknown
education-spending:y,n,unknown
superfund-right-to-sue:y,n,unknown
crime:y,n,unknown
duty-free-exports:y,n,unknown
export-administration-act-south-africa:y,n,unknown
class:democrat,republican
```

### .data/.train/.test File Format
Comma-separated values matching the attributes defined in .meta file:
```
n,y,n,y,y,y,n,n,n,y,unknown,y,y,y,n,y,republican
n,y,n,y,y,y,n,n,n,n,n,y,y,y,n,unknown,republican
unknown,y,y,unknown,y,y,n,n,n,n,y,n,y,y,n,n,democrat
y,y,y,n,n,n,y,y,y,n,n,n,y,n,y,y,democrat
```

**Note:** Class label must be in the last column. Missing values are represented as `unknown`.

## Usage Examples

### Example 1: K-Fold Cross Validation
```bash
python3 main.py --mode kfold --data voting.data --meta voting.meta --k 5
```

**Output:**
```
Accuracy score for fold 1: 92.05
Metrics for fold 1: 
democrat --- Precision: 0.91 Recall: 0.95 Specificity: 0.89 f1: 0.93
republican --- Precision: 0.94 Recall: 0.88 Specificity: 0.95 f1: 0.91
...

Average accuracy: 91.26
```

### Example 2: Train and Save Model
```bash
python3 main.py --mode train --data voting.train --meta voting.meta --model voting_classifier.pkl
```

**Output:**
```
Loading training data from voting.train...
Training on 304 instances
Training completed!
Model saved to voting_classifier.pkl

Model Statistics:
Classes: ['democrat', 'republican']
Attributes: ['handicapped-infants', 'water-project-cost-sharing', ...]
Total training instances: 304
```

### Example 3: Predict with Saved Model
```bash
python3 main.py --mode predict --input voting.test --output predictions.csv --model voting_classifier.pkl
```

**Output:**
```
Model loaded from voting_classifier.pkl
Classifying 131 instances...
Predictions saved to predictions.csv
Classified 131 instances
```

### Example 4: Train and Evaluate
```bash
python3 main.py --mode evaluate --train voting.train --test voting.test --meta voting.meta
```

**Output:**
```
Loading training data from voting.train...
Training on 304 instances...
Training completed!

Loading test data from voting.test...
Evaluating on 131 instances...

==================================================
EVALUATION RESULTS
==================================================
Accuracy Score: 93.89%
Error Rate: 6.11%

Confusion Matrix
Actual/Predicted  democrat  republican
democrat              78          9
republican             5         39

==================================================
DETAILED METRICS
==================================================
democrat:
  Precision:   0.9398
  Recall:      0.8966
  Specificity: 0.8125
  F1-Score:    0.9177

republican:
  Precision:   0.8125
  Recall:      0.8864
  Specificity: 0.8966
  F1-Score:    0.8479
```

## Algorithm Details

### Naive Bayes with Laplace Smoothing

The classifier uses Laplace smoothing (add-one smoothing) to handle zero probabilities:
```
P(value|class) = (count(value, class) + 1) / (count(class) + num_possible_values)
```

### Classification Decision

For each instance, the classifier:
1. Calculates prior probability: `P(class)`
2. Multiplies by conditional probabilities: `P(attribute1|class) × P(attribute2|class) × ...`
3. Selects the class with highest posterior probability

### Missing Data Handling

Missing values (represented as `?` in the original data) are converted to `unknown` and treated as a separate categorical value. This allows the classifier to learn patterns associated with missing data rather than discarding valuable instances.

## Performance Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **Specificity**: True Negatives / (True Negatives + False Positives)
- **F1-Score**: Harmonic mean of Precision and Recall

## Dataset

This classifier is tested with the Congressional Voting Records dataset and supports various categorical datasets from the UCI Machine Learning Repository.

### Congressional Voting Records (Included)
- **Citation**: Schlimmer, J. (1987). Congressional Voting Records [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5C01P
- **Description**: 1984 United States Congressional Voting Records. Classify congressperson as Democrat or Republican based on their votes on 16 key issues
- **Size**: 435 instances (one per congressperson), 16 attributes, 2 classes
- **Missing Data**: 66.2% of instances contain at least one missing vote (handled as 'unknown')
- **Expected Accuracy**: ~90-95% with k-fold cross-validation
- **Setup**: Run `python3 convert_voting.py` to download and format the dataset

## Project Structure
```
.
├── main.py                # Main classifier implementation
├── convert_voting.py      # Script to download and convert voting dataset
├── README.md              # This file
├── voting.data            # Congressional voting records dataset
├── voting.meta            # Metadata for voting dataset
├── voting.train           # Training split (70%)
├── voting.test            # Test split (30%)
└── nbc_model.pkl          # Example saved model (after training)
```

## Getting Started with the Voting Dataset

1. **Download and convert the dataset:**
```bash
python3 convert_voting.py
```

2. **Run k-fold cross-validation:**
```bash
python3 main.py --mode kfold --data voting.data --meta voting.meta --k 5
```

3. **Train and evaluate:**
```bash
python3 main.py --mode evaluate --train voting.train --test voting.test --meta voting.meta
```






