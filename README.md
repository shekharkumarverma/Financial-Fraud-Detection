# Financial-Fraud-Detection
Project Overview
This project focuses on detecting fraudulent transactions using machine learning techniques. The dataset used is from Kaggle (BankSim1 dataset), containing transaction records with various attributes like customer age, gender, ZIP code, merchant, purchase category, amount, and fraud labels (0/1).

Team Members
Justin Gould

Baylee Adams

Kyle Shope

Project Goals
Apply clustering and classification algorithms to distinguish between legitimate and fraudulent transactions.

Identify anomalous transactions that deviate from typical customer purchase behavior.

Dataset
The dataset includes:

Transaction attributes (customer age, gender, ZIP code, merchant, purchase category, amount, etc.)

Fraud labels (0 for legitimate, 1 for fraudulent)

Requirements
Python Packages
pandas

numpy

scikit-learn

matplotlib

seaborn

graphviz

pydot

pydotplus

Code Structure
Data Loading and Preprocessing

Load original, upsampled, and test datasets.

Clean and preprocess data (remove apostrophes, adjust column data types, drop unnecessary columns).

Convert categorical columns to binary columns using one-hot encoding.

Exploratory Data Analysis

Visualize the distribution of transaction amounts.

Standardize the data for clustering.

Customer Clustering

Use K-Means clustering to group customers based on purchase behavior.

Determine the optimal number of clusters using the elbow method.

Analyze the distribution of fraud within each cluster.

Model Training and Evaluation

Split data into training and test sets.

Train classification models (e.g., KNN, Naive Bayes, Logistic Regression, Decision Trees, SVM).

Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

Key Findings
Clustering Results: Customers were grouped into 4 clusters, with varying fraud rates:

Cluster 0: 1.31% fraud

Cluster 1: 25% fraud

Cluster 2: 0.8% fraud

Cluster 3: 79.4% fraud

Model Performance: The best-performing model achieved high accuracy in distinguishing fraudulent transactions.

How to Run
Install the required packages:

bash
pip install pandas numpy scikit-learn matplotlib seaborn graphviz pydot pydotplus
Download the dataset from Kaggle and place it in the project directory.

Run the Jupyter notebook (Fraud_Project.ipynb) to execute the analysis.

Future Work
Experiment with more advanced models (e.g., Random Forest, Neural Networks).

Incorporate additional features or external data sources to improve detection accuracy.

Deploy the model as a real-time fraud detection system.

Acknowledgments
Dataset sourced from Kaggle: BankSim1 Dataset

Special thanks to Michigan State University for support.
