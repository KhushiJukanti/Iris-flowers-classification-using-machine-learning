# Iris-flowers-classification-using-machine-learning


This repository contains the code and resources for classifying Iris flowers using machine learning algorithms. The project focuses on analyzing various features of Iris flowers and training a model to accurately classify them into different species.

## Dataset

The dataset used for this project is the [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) from the UCI Machine Learning Repository. It consists of 150 samples of Iris flowers, with each sample having measurements of sepal length, sepal width, petal length, and petal width. The dataset is available in the `data` directory.

## Dependencies

The following dependencies are required to run the code in this repository:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Code Structure

- `data/`: Directory containing the dataset file.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- `src/`: Python scripts for preprocessing, model training, and evaluation.
- `models/`: Directory to save trained models.
- `utils/`: Utility functions and classes used in the project.
- `README.md`: This file, providing an overview of the repository.

## Usage

To train and evaluate the Iris flower classification model, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/iris-flowers-classification.git
   cd iris-flowers-classification
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Explore the Jupyter notebooks in the `notebooks/` directory to understand the project workflow and data analysis.

4. Preprocess the data:

   ```bash
   python src/data_preprocessing.py
   ```

   This script performs data cleaning and splitting the dataset into training and testing sets.

5. Train the classification model:

   ```bash
   python src/train_model.py
   ```

   This script trains a machine learning model on the preprocessed data and saves it in the `models/` directory.

6. Evaluate the model:

   ```bash
   python src/evaluate_model.py
   ```

   This script evaluates the trained model using various metrics.

## Results

The results of the classification model, including evaluation metrics, can be found in the `results/` directory.

## Contributing

Contributions to this repository are always welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

