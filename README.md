Machine Learning Model Comparison App
A Shiny application for training and comparing different machine learning models for classification tasks.
Features

Data import and preprocessing
Model training and evaluation:

Logistic Regression
K-Nearest Neighbors
Naive Bayes
Decision Tree
Random Forest


Model comparison tools:

Accuracy comparison
ROC curve visualization
Confusion matrix visualization
Performance metrics (Accuracy, Sensitivity, Specificity, AUC)



Requirements

R (>= 3.6.0)
Required R packages:

shiny
DT
ggplot2
caret
randomForest
rpart
rpart.plot
pROC
e1071
class



Installation

Clone this repository
Install the required packages:

rinstall.packages(c("shiny", "DT", "ggplot2", "caret", "randomForest", 
                  "rpart", "rpart.plot", "pROC", "e1071", "class"))

Run the app:

rshiny::runApp()
Usage

Upload your dataset
Configure settings for each model
Train models
Compare model performance
