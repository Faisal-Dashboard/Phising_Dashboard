install.packages("tidyverse")
install.packages("mice")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("Amelia")
install.packages("FNN")
install.packages("e1071")
install.packages("rpart")
install.packages("randomForest")
install.packages("ROCR")
install.packages("pROC")
install.packages("caret")
install.packages("rpart.plot")

# Load necessary libraries
library(tidyverse)  # For data manipulation and visualization
library(mice)       # For handling missing values
library(corrplot)   # For correlation matrix visualization
library(ggplot2)    # For plotting
library(Amelia)     # For visualizing missing values
library(class)
library(FNN)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)    
library(pROC)
library(caret)

# Read the dataset
data <- read.csv("Phishing_Legitimate.csv", na.strings = c("", "NA"))

# View basic structure
str(data)
summary(data)

# Check missing values per column
missing_values <- colSums(is.na(data))
print(missing_values)

# Calculate percentage of missing values
missing_percentage <- (missing_values / nrow(data)) * 100
print(missing_percentage)

# Decision to handle missing values
if (max(missing_percentage) > 30) {
  data <- data[, which(missing_percentage <= 30)]  # Drop columns with >30% missing data
  cat("Dropped columns with more than 30% missing values.\n")
} else if (mean(missing_percentage) > 1) {
  data <- complete(mice(data, method = "pmm", m = 5))  # Impute if avg missing >1%
  cat("Missing values imputed using mice().\n")
} else {
  data <- na.omit(data)  # Remove rows with missing values if <1% overall
  cat("Rows with missing values removed.\n")
}

cat("Final dataset dimensions:", dim(data), "\n")

# Boxplot to visualize outliers
boxplot(data, main = "Boxplot for Outlier Detection", col = "lightblue")

# Find outliers using IQR method
Q1 <- apply(data, 2, quantile, 0.25, na.rm = TRUE)
Q3 <- apply(data, 2, quantile, 0.75, na.rm = TRUE)
IQR_values <- Q3 - Q1

# Define outlier limits
lower_bound <- Q1 - 1.5 * IQR_values
upper_bound <- Q3 + 1.5 * IQR_values

outliers <- data < lower_bound | data > upper_bound

cat("Total outliers identified:", sum(outliers, na.rm = TRUE), "\n")

# Calculate correlation matrix
cor_matrix <- cor(data, use = "pairwise.complete.obs")

# Visualize correlation
corrplot(cor_matrix, method = "color", tl.cex = 0.7)

# Identify constant (low variance) variables
low_variance <- apply(data, 2, var, na.rm = TRUE) == 0
data <- data[, !low_variance]

# Check unique values per column
unique_values <- sapply(data, function(x) length(unique(x)))
print(unique_values)

# Remove variables with very few unique values (e.g., identifier columns)
data <- data[, unique_values > 1]
cat("Low-variance and identifier columns removed.\n")

# Min-Max Scaling Function
min_max_scaling <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

# Apply Min-Max Scaling to Numeric Columns
for (col in names(data)) {
  if (is.numeric(data[[col]])) {
    data[[col]] <- min_max_scaling(data[[col]])
  }
}

# Read the original dataset for visualization
df <- read.csv("Phishing_Legitimate.csv", na.strings = c("", "NA"))

# Check the structure of the dataset to confirm column names
str(df)

# Exploratory Data Analysis
ggplot(df, aes(x = as.factor(CLASS_LABEL))) + 
  geom_bar(fill = "blue") + 
  labs(title = "Distribution of Phishing vs Legitimate Websites",
       x = "Class (1 = Phishing, 0 = Legitimate)", 
       y = "Count")

ggplot(df, aes(x = UrlLength)) + 
  geom_histogram(fill = "red", bins = 30, alpha = 0.7) + 
  labs(title = "Distribution of URL Length", x = "URL Length", y = "Frequency")

ggplot(df, aes(x = as.factor(CLASS_LABEL), y = UrlLength)) + 
  geom_boxplot(fill = "orange") + 
  labs(title = "Boxplot of URL Length by Class",
       x = "Class (1 = Phishing, 0 = Legitimate)", y = "URL Length")

ggplot(df, aes(x = UrlLength, fill = as.factor(CLASS_LABEL))) + 
  geom_density(alpha = 0.6) + 
  labs(title = "Density Plot of URL Length", x = "URL Length", y = "Density")

#--------------------MODEL BUILDING--------------------#

# Clean the dataset: Handle missing values, factorize the target variable
data_cleaned <- na.omit(data)  # Remove rows with missing values, or you could use imputation
data_cleaned$CLASS_LABEL <- as.factor(data_cleaned$CLASS_LABEL)

# Data Partitioning (80% training, 20% testing)
set.seed(123)
train_index <- sample(1:nrow(data_cleaned), 0.8 * nrow(data_cleaned))
train_data <- data_cleaned[train_index, ]
test_data <- data_cleaned[-train_index, ]

#--------------------LOGISTIC REGRESSION--------------------#
# Model Training: Logistic Regression
logistic_model <- glm(CLASS_LABEL ~ ., data = train_data, family = "binomial")

# Model Testing: Predict using the trained model
pred_logistic <- predict(logistic_model, test_data, type = "response")
pred_logistic_class <- ifelse(pred_logistic > 0.5, 1, 0)

# Model Evaluation: Confusion Matrix
conf_matrix_logistic <- table(Predicted = pred_logistic_class, Actual = test_data$CLASS_LABEL)
print("Logistic Regression Confusion Matrix:")
print(conf_matrix_logistic)

# Model Evaluation metrics: Accuracy
accuracy_logistic <- sum(diag(conf_matrix_logistic)) / sum(conf_matrix_logistic)
print(paste("Logistic Regression Accuracy:", accuracy_logistic))

# ROC Curve: Performance Evaluation
roc_curve <- roc(test_data$CLASS_LABEL, pred_logistic)
plot(roc_curve, main = "ROC Curve for Logistic Regression")
print(paste("Logistic Regression AUC:", auc(roc_curve)))

#--------------------KNN--------------------#
# Data Preparation for KNN
train_data_knn <- train_data[, -which(names(train_data) == "CLASS_LABEL")]
test_data_knn <- test_data[, -which(names(test_data) == "CLASS_LABEL")]
train_label_knn <- train_data$CLASS_LABEL
test_label_knn <- test_data$CLASS_LABEL

# Model Training for KNN
k_value <- 5
knn_model <- knn(train = train_data_knn, test = test_data_knn, cl = train_label_knn, k = k_value)

# Model Evaluation for KNN
conf_matrix_knn <- table(Predicted = knn_model, Actual = test_label_knn)
print("KNN Confusion Matrix:")
print(conf_matrix_knn)

accuracy_knn <- sum(diag(conf_matrix_knn)) / sum(conf_matrix_knn)
print(paste("KNN Accuracy:", accuracy_knn))

#--------------------NAIVE BAYES--------------------#
# Model Training for Naive Bayes
nb_model <- naiveBayes(CLASS_LABEL ~ ., data = train_data)

# Model Testing for Naive Bayes
pred_nb <- predict(nb_model, test_data)
pred_prob_nb <- predict(nb_model, test_data, type = "raw")

# Model Evaluation for Naive Bayes
conf_matrix_nb <- table(Predicted = pred_nb, Actual = test_data$CLASS_LABEL)
print("Naive Bayes Confusion Matrix:")
print(conf_matrix_nb)

accuracy_nb <- sum(diag(conf_matrix_nb)) / sum(conf_matrix_nb)
print(paste("Naive Bayes Accuracy:", accuracy_nb))

# ROC Curve for Naive Bayes
pred_obj_nb <- prediction(pred_prob_nb[,2], as.numeric(as.character(test_data$CLASS_LABEL)))
roc_perf_nb <- performance(pred_obj_nb, "tpr", "fpr")
auc_perf_nb <- performance(pred_obj_nb, "auc")

plot(roc_perf_nb,
     main = "ROC Curve for Naïve Bayes Model",
     col = "#2c7bb6",
     lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")
print(paste("Naive Bayes AUC:", round(auc_perf_nb@y.values[[1]], 4)))

#--------------------DECISION TREE--------------------#
# Model Training for Decision Tree
dt_model <- rpart(CLASS_LABEL ~ ., data = train_data, method = "class")

# Model Testing for Decision Tree
pred_dt <- predict(dt_model, test_data, type = "class")

# Model Evaluation for Decision Tree
conf_matrix_dt <- table(Predicted = pred_dt, Actual = test_data$CLASS_LABEL)
print("Decision Tree Confusion Matrix:")
print(conf_matrix_dt)

accuracy_dt <- sum(diag(conf_matrix_dt)) / sum(conf_matrix_dt)
print(paste("Decision Tree Accuracy:", accuracy_dt))

# Visualize Decision Tree
rpart.plot(dt_model, main = "Decision Tree")

#--------------------RANDOM FOREST--------------------#
# Model Training for Random Forest
rf_model <- randomForest(CLASS_LABEL ~ ., 
                         data = train_data,
                         ntree = 100,
                         importance = TRUE)

# Model Testing for Random Forest
pred_rf <- predict(rf_model, test_data)
pred_prob_rf <- predict(rf_model, test_data, type = "prob")

# Model Evaluation for Random Forest
conf_matrix_rf <- table(Predicted = pred_rf, Actual = test_data$CLASS_LABEL)
print("Random Forest Confusion Matrix:")
print(conf_matrix_rf)

accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
print(paste("Random Forest Accuracy:", accuracy_rf))

# ROC Curve for Random Forest
pred_obj_rf <- prediction(pred_prob_rf[,2], as.numeric(as.character(test_data$CLASS_LABEL)))
roc_perf_rf <- performance(pred_obj_rf, "tpr", "fpr")
auc_perf_rf <- performance(pred_obj_rf, "auc")

plot(roc_perf_rf,
     main = "ROC Curve for Random Forest Model",
     col = "#4daf4a",
     lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")
print(paste("Random Forest AUC:", round(auc_perf_rf@y.values[[1]], 4)))

# Feature Importance Plot for Random Forest
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "MeanDecreaseGini"]
) %>% 
  arrange(desc(Importance)) %>% 
  head(20)  # Show top 20 features

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "#4daf4a", alpha = 0.8) +
  coord_flip() +
  labs(title = "Random Forest - Feature Importance (Gini Index)",
       x = "Features",
       y = "Mean Decrease in Gini Index") +
  theme_minimal()

#--------------------MODEL COMPARISON--------------------#
# Create data frame with models and accuracy values
results <- data.frame(
  Model = c("Logistic Regression", "KNN", "Naïve Bayes", "Decision Tree", "Random Forest"),
  Accuracy = c(accuracy_logistic, accuracy_knn, accuracy_nb, accuracy_dt, accuracy_rf)
)

# Print the comparison results
print(results)

# Create bar chart for model comparison
ggplot(results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")), 
            vjust = -0.3, size = 4) +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Function to extract all performance metrics from confusion matrix
get_metrics <- function(predicted, actual) {
  predicted <- factor(predicted, levels = levels(actual))
  cm <- confusionMatrix(predicted, actual, positive = "1")
  return(data.frame(
    Accuracy = cm$overall['Accuracy'],
    Sensitivity = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity'],
    Precision = cm$byClass['Pos Pred Value'],
    F1_Score = cm$byClass['F1']
  ))
}

# Collect metrics for all models
metrics_logistic <- get_metrics(factor(pred_logistic_class), test_data$CLASS_LABEL)
metrics_knn <- get_metrics(knn_model, test_label_knn)
metrics_nb <- get_metrics(pred_nb, test_data$CLASS_LABEL)
metrics_dt <- get_metrics(pred_dt, test_data$CLASS_LABEL)
metrics_rf <- get_metrics(pred_rf, test_data$CLASS_LABEL)

# Combine all results into one final table
model_results <- rbind(
  cbind(Model = "Logistic Regression", metrics_logistic),
  cbind(Model = "KNN", metrics_knn),
  cbind(Model = "Naive Bayes", metrics_nb),
  cbind(Model = "Decision Tree", metrics_dt),
  cbind(Model = "Random Forest", metrics_rf)
)

# View the final comparison table
print(model_results)
