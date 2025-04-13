# Install required packages if not already installed
if (!require("shiny")) install.packages("shiny")
if (!require("shinydashboard")) install.packages("shinydashboard")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("mice")) install.packages("mice")
if (!require("corrplot")) install.packages("corrplot")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("Amelia")) install.packages("Amelia")
if (!require("FNN")) install.packages("FNN")
if (!require("e1071")) install.packages("e1071")
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("randomForest")) install.packages("randomForest")
if (!require("ROCR")) install.packages("ROCR")
if (!require("pROC")) install.packages("pROC")
if (!require("DT")) install.packages("DT")
if (!require("caret")) install.packages("caret")
if (!require("class")) install.packages("class")
if (!require("reshape2")) install.packages("reshape2")

# Load required libraries
library(shiny)
library(shinydashboard)
library(tidyverse)
library(mice)
library(corrplot)
library(ggplot2)
library(Amelia)
library(class)
library(FNN)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(pROC)
library(DT)
library(caret)
library(reshape2)

# UI Definition
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Phishing Detection"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dataset", tabName = "dataset", icon = icon("database")),
      menuItem("Data Preprocessing", tabName = "preprocessing", icon = icon("wrench")),
      menuItem("Exploratory Analysis", tabName = "eda", icon = icon("chart-simple")),
      menuItem("Model Training", tabName = "modeling", icon = icon("gears")),
      menuItem("Model Comparison", tabName = "comparison", icon = icon("table")),
      menuItem("About", tabName = "about", icon = icon("circle-info"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # Dataset Tab
      tabItem(tabName = "dataset",
              fluidRow(
                box(
                  title = "Upload Dataset", status = "primary", solidHeader = TRUE, width = 12,
                  fileInput("file", "Upload CSV File (or use sample data)", accept = c(".csv")),
                  actionButton("use_sample", "Use Sample Data"),
                  br(), br(),
                  checkboxInput("header", "Header", TRUE),
                  radioButtons("sep", "Separator",
                               choices = c(Comma = ",", Semicolon = ";", Tab = "\t"),
                               selected = ",")
                )
              ),
              fluidRow(
                tabBox(
                  title = "Dataset Overview", width = 12,
                  tabPanel("Data Preview", DT::dataTableOutput("data_preview")),
                  tabPanel("Structure", verbatimTextOutput("structure")),
                  tabPanel("Summary", verbatimTextOutput("summary"))
                )
              )
      ),
      
      # Data Preprocessing Tab
      tabItem(tabName = "preprocessing",
              fluidRow(
                box(
                  title = "Missing Values", status = "primary", solidHeader = TRUE, width = 6,
                  plotOutput("missing_plot"),
                  verbatimTextOutput("missing_summary")
                ),
                box(
                  title = "Handle Missing Values", status = "primary", solidHeader = TRUE, width = 6,
                  radioButtons("missing_method", "Method:",
                               choices = c("Remove rows" = "remove",
                                           "Remove columns with >30% missing" = "remove_cols",
                                           "Impute using MICE" = "mice"),
                               selected = "mice"),
                  actionButton("handle_missing", "Apply"),
                  br(), br(),
                  verbatimTextOutput("missing_result")
                )
              ),
              fluidRow(
                box(
                  title = "Outliers", status = "primary", solidHeader = TRUE, width = 6,
                  plotOutput("outlier_plot"),
                  verbatimTextOutput("outlier_summary")
                ),
                box(
                  title = "Feature Engineering", status = "primary", solidHeader = TRUE, width = 6,
                  checkboxInput("remove_low_var", "Remove Low Variance Features", TRUE),
                  checkboxInput("min_max_scale", "Apply Min-Max Scaling", TRUE),
                  actionButton("preprocess", "Apply Preprocessing"),
                  br(), br(),
                  verbatimTextOutput("preprocess_result")
                )
              )
      ),
      
      # Exploratory Data Analysis Tab
      tabItem(tabName = "eda",
              fluidRow(
                box(
                  title = "Class Distribution", status = "primary", solidHeader = TRUE, width = 6,
                  plotOutput("class_dist_plot")
                ),
                box(
                  title = "URL Length Distribution", status = "primary", solidHeader = TRUE, width = 6,
                  plotOutput("url_length_plot")
                )
              ),
              fluidRow(
                box(
                  title = "Feature Correlation", status = "primary", solidHeader = TRUE, width = 12,
                  sliderInput("corr_cutoff", "Correlation Cutoff:", min = 0, max = 1, value = 0.7, step = 0.05),
                  plotOutput("correlation_plot", height = "600px")
                )
              ),
              fluidRow(
                box(
                  title = "URL Length by Class", status = "primary", solidHeader = TRUE, width = 6,
                  plotOutput("url_box_plot")
                ),
                box(
                  title = "URL Length Density", status = "primary", solidHeader = TRUE, width = 6,
                  plotOutput("url_density_plot")
                )
              )
      ),
      
      # Model Training Tab
      tabItem(tabName = "modeling",
              fluidRow(
                box(
                  title = "Training Settings", status = "primary", solidHeader = TRUE, width = 12,
                  sliderInput("train_test_split", "Training Data %:", min = 50, max = 90, value = 80, step = 5),
                  numericInput("random_seed", "Random Seed:", value = 123),
                  actionButton("split_data", "Split Data")
                )
              ),
              fluidRow(
                tabBox(
                  title = "Models", width = 12,
                  tabPanel("Logistic Regression",
                           actionButton("train_logistic", "Train Logistic Regression"),
                           plotOutput("logistic_roc"),
                           verbatimTextOutput("logistic_summary")),
                  tabPanel("KNN",
                           numericInput("knn_k", "Number of Neighbors (k):", value = 5, min = 1, max = 50),
                           actionButton("train_knn", "Train KNN"),
                           plotOutput("knn_plot"),
                           verbatimTextOutput("knn_summary")),
                  tabPanel("Naive Bayes",
                           actionButton("train_nb", "Train Naive Bayes"),
                           plotOutput("nb_roc"),
                           verbatimTextOutput("nb_summary")),
                  tabPanel("Decision Tree",
                           numericInput("dt_depth", "Maximum Tree Depth:", value = 10, min = 1, max = 30),
                           actionButton("train_dt", "Train Decision Tree"),
                           plotOutput("dt_plot"),
                           verbatimTextOutput("dt_summary")),
                  tabPanel("Random Forest",
                           numericInput("rf_trees", "Number of Trees:", value = 100, min = 10, max = 500),
                           actionButton("train_rf", "Train Random Forest"),
                           plotOutput("rf_importance"),
                           verbatimTextOutput("rf_summary"))
                )
              )
      ),
      
      # Model Comparison Tab
      tabItem(tabName = "comparison",
              fluidRow(
                box(
                  title = "Accuracy Comparison", status = "primary", solidHeader = TRUE, width = 6,
                  plotOutput("accuracy_plot")
                ),
                box(
                  title = "ROC Curves Comparison", status = "primary", solidHeader = TRUE, width = 6,
                  plotOutput("roc_comparison")
                )
              ),
              fluidRow(
                box(
                  title = "Detailed Metrics", status = "primary", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("metrics_table")
                )
              ),
              fluidRow(
                box(
                  title = "Confusion Matrices", status = "primary", solidHeader = TRUE, width = 12,
                  selectInput("cm_model", "Select Model:", 
                              choices = c("Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "Random Forest")),
                  plotOutput("confusion_matrix")
                )
              )
      ),
      
      # About Tab
      tabItem(tabName = "about",
              fluidRow(
                box(
                  title = "About This Dashboard", status = "primary", solidHeader = TRUE, width = 12,
                  h3("Phishing Website Detection Dashboard"),
                  p("This dashboard provides tools for analyzing and comparing machine learning models for phishing website detection."),
                  p("Features include:"),
                  tags$ul(
                    tags$li("Data loading and preprocessing"),
                    tags$li("Exploratory data analysis"),
                    tags$li("Model training and evaluation"),
                    tags$li("Model comparison")
                  ),
                  h4("Models implemented:"),
                  tags$ul(
                    tags$li("Logistic Regression"),
                    tags$li("K-Nearest Neighbors (KNN)"),
                    tags$li("Naive Bayes"),
                    tags$li("Decision Tree"),
                    tags$li("Random Forest")
                  )
                )
              )
      )
    )
  )
)

# Server Logic
server <- function(input, output, session) {
  # Reactive values to store data and models
  values <- reactiveValues(
    data = NULL,
    data_clean = NULL,
    train_data = NULL,
    test_data = NULL,
    logistic_model = NULL,
    knn_model = NULL,
    nb_model = NULL,
    dt_model = NULL,
    rf_model = NULL,
    metrics = data.frame(
      Model = character(),
      Accuracy = numeric(),
      Sensitivity = numeric(),
      Specificity = numeric(),
      AUC = numeric(),
      stringsAsFactors = FALSE
    )
  )
  
  # Sample data for demonstration
  observeEvent(input$use_sample, {
    values$data <- read.csv("https://raw.githubusercontent.com/datasets-io/phishing-websites/master/data.csv")
    # If the sample URL doesn't work, you can create mock data
    if (is.null(values$data) || nrow(values$data) == 0) {
      set.seed(123)
      n <- 1000
      values$data <- data.frame(
        UrlLength = sample(20:200, n, replace = TRUE),
        NumDots = sample(0:10, n, replace = TRUE),
        NumDash = sample(0:8, n, replace = TRUE),
        NumUnderscore = sample(0:5, n, replace = TRUE),
        NumPercent = sample(0:4, n, replace = TRUE),
        NumAmpersand = sample(0:3, n, replace = TRUE),
        NumSlash = sample(1:15, n, replace = TRUE),
        HasHttps = sample(0:1, n, replace = TRUE),
        HasIP = sample(0:1, n, replace = TRUE),
        DomainLength = sample(5:50, n, replace = TRUE),
        NumSubdomains = sample(0:5, n, replace = TRUE),
        PathLength = sample(0:100, n, replace = TRUE),
        QueryLength = sample(0:150, n, replace = TRUE),
        NumParameters = sample(0:10, n, replace = TRUE),
        CLASS_LABEL = sample(0:1, n, replace = TRUE, prob = c(0.6, 0.4))
      )
    }
  })
  
  # Load data when file is uploaded
  observeEvent(input$file, {
    req(input$file)
    values$data <- read.csv(input$file$datapath, 
                            header = input$header,
                            sep = input$sep,
                            na.strings = c("", "NA"))
  })
  
  # Data Preview
  output$data_preview <- DT::renderDataTable({
    req(values$data)
    DT::datatable(values$data, options = list(scrollX = TRUE))
  })
  
  # Structure
  output$structure <- renderPrint({
    req(values$data)
    str(values$data)
  })
  
  # Summary
  output$summary <- renderPrint({
    req(values$data)
    summary(values$data)
  })
  
  # Missing Values Plot
  output$missing_plot <- renderPlot({
    req(values$data)
    if(sum(is.na(values$data)) > 0) {
      missmap(values$data, main = "Missing Values Map", 
              col = c("red", "steelblue"), legend = TRUE)
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "No missing values found", cex = 1.5)
    }
  })
  
  # Missing Values Summary
  output$missing_summary <- renderPrint({
    req(values$data)
    missing_values <- colSums(is.na(values$data))
    missing_percentage <- (missing_values / nrow(values$data)) * 100
    
    cat("Missing Values Summary:\n")
    for(col in names(values$data)) {
      if(missing_values[col] > 0) {
        cat(paste0(col, ": ", missing_values[col], " (", 
                   round(missing_percentage[col], 2), "%)\n"))
      }
    }
    
    if(sum(missing_values) == 0) {
      cat("No missing values found in the dataset.")
    } else {
      cat("\nTotal missing values: ", sum(missing_values), " (", 
          round(sum(missing_values)/(nrow(values$data)*ncol(values$data))*100, 2), "%)")
    }
  })
  
  # Handle Missing Values
  observeEvent(input$handle_missing, {
    req(values$data)
    data <- values$data
    
    if(sum(is.na(data)) == 0) {
      values$data_clean <- data
      output$missing_result <- renderText("No missing values to handle.")
      return()
    }
    
    missing_percentage <- colSums(is.na(data)) / nrow(data) * 100
    
    if(input$missing_method == "remove") {
      data <- na.omit(data)
      output$missing_result <- renderText(paste0("Removed ", nrow(values$data) - nrow(data), " rows with missing values."))
    } else if(input$missing_method == "remove_cols") {
      cols_to_keep <- names(data)[missing_percentage <= 30]
      data <- data[, cols_to_keep]
      output$missing_result <- renderText(paste0("Removed ", ncol(values$data) - ncol(data), " columns with >30% missing values."))
    } else if(input$missing_method == "mice") {
      if(any(missing_percentage > 0)) {
        withProgress(message = 'Imputing missing values...', {
          imp <- mice(data, method = "pmm", m = 1, maxit = 5, printFlag = FALSE)
          data <- complete(imp)
        })
        output$missing_result <- renderText("Missing values imputed using MICE (Multivariate Imputation by Chained Equations).")
      }
    }
    
    values$data_clean <- data
  })
  
  # Outlier Plot
  output$outlier_plot <- renderPlot({
    req(values$data_clean)
    
    # Select only numeric columns for boxplot
    numeric_cols <- sapply(values$data_clean, is.numeric)
    if(sum(numeric_cols) > 0) {
      boxplot(values$data_clean[, numeric_cols], 
              main = "Boxplot for Outlier Detection", 
              col = "lightblue",
              las = 2)
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "No numeric columns for outlier detection", cex = 1.5)
    }
  })
  
  # Outlier Summary
  output$outlier_summary <- renderPrint({
    req(values$data_clean)
    
    # Select only numeric columns
    numeric_data <- values$data_clean[, sapply(values$data_clean, is.numeric)]
    
    if(ncol(numeric_data) == 0) {
      cat("No numeric columns for outlier detection.")
      return()
    }
    
    # Function to detect outliers using IQR method
    detect_outliers <- function(x) {
      if(length(unique(x)) <= 1) return(0)
      q1 <- quantile(x, 0.25, na.rm = TRUE)
      q3 <- quantile(x, 0.75, na.rm = TRUE)
      iqr <- q3 - q1
      lower_bound <- q1 - 1.5 * iqr
      upper_bound <- q3 + 1.5 * iqr
      sum(x < lower_bound | x > upper_bound, na.rm = TRUE)
    }
    
    outliers_count <- sapply(numeric_data, detect_outliers)
    outliers_percentage <- (outliers_count / nrow(numeric_data)) * 100
    
    cat("Outliers Summary:\n")
    for(col in names(outliers_count)) {
      if(outliers_count[col] > 0) {
        cat(paste0(col, ": ", outliers_count[col], " (", 
                   round(outliers_percentage[col], 2), "%)\n"))
      }
    }
    
    if(sum(outliers_count) == 0) {
      cat("No outliers detected in the dataset.")
    } else {
      cat("\nTotal outliers across all numeric columns: ", sum(outliers_count))
    }
  })
  
  # Apply Preprocessing
  observeEvent(input$preprocess, {
    req(values$data_clean)
    data <- values$data_clean
    
    # Remove low variance features if requested
    if(input$remove_low_var) {
      numeric_cols <- sapply(data, is.numeric)
      if(sum(numeric_cols) > 0) {
        vars <- apply(data[, numeric_cols], 2, var, na.rm = TRUE)
        low_var <- vars < 0.01 | is.na(vars)
        if(any(low_var)) {
          data <- data[, !low_var | !numeric_cols]
        }
      }
    }
    
    # Apply Min-Max scaling to numeric columns if requested
    if(input$min_max_scale) {
      numeric_cols <- sapply(data, is.numeric)
      if(sum(numeric_cols) > 0) {
        min_max_scale <- function(x) {
          if(length(unique(x)) <= 1) return(x)
          (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
        }
        
        data[, numeric_cols] <- lapply(data[, numeric_cols], min_max_scale)
      }
    }
    
    values$data_clean <- data
    
    # Create preprocessing result message
    msg <- "Preprocessing applied:\n"
    if(input$remove_low_var) {
      msg <- paste0(msg, "- Low variance features removed\n")
    }
    if(input$min_max_scale) {
      msg <- paste0(msg, "- Min-Max scaling applied to numeric features\n")
    }
    
    output$preprocess_result <- renderText(msg)
  })
  
  # Class Distribution Plot
  output$class_dist_plot <- renderPlot({
    req(values$data_clean)
    
    if("CLASS_LABEL" %in% names(values$data_clean)) {
      ggplot(values$data_clean, aes(x = as.factor(CLASS_LABEL))) + 
        geom_bar(fill = "blue") + 
        labs(title = "Distribution of Phishing vs Legitimate Websites",
             x = "Class (1 = Phishing, 0 = Legitimate)", 
             y = "Count") +
        theme_minimal()
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "CLASS_LABEL column not found", cex = 1.5)
    }
  })
  
  # URL Length Distribution Plot
  output$url_length_plot <- renderPlot({
    req(values$data_clean)
    
    if("UrlLength" %in% names(values$data_clean)) {
      ggplot(values$data_clean, aes(x = UrlLength)) + 
        geom_histogram(fill = "red", bins = 30, alpha = 0.7) + 
        labs(title = "Distribution of URL Length", x = "URL Length", y = "Frequency") +
        theme_minimal()
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "UrlLength column not found", cex = 1.5)
    }
  })
  
  # URL Length Boxplot by Class
  output$url_box_plot <- renderPlot({
    req(values$data_clean)
    
    if(all(c("UrlLength", "CLASS_LABEL") %in% names(values$data_clean))) {
      ggplot(values$data_clean, aes(x = as.factor(CLASS_LABEL), y = UrlLength)) + 
        geom_boxplot(fill = "orange") + 
        labs(title = "Boxplot of URL Length by Class",
             x = "Class (1 = Phishing, 0 = Legitimate)", y = "URL Length") +
        theme_minimal()
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "Required columns not found", cex = 1.5)
    }
  })
  
  # URL Length Density Plot
  output$url_density_plot <- renderPlot({
    req(values$data_clean)
    
    if(all(c("UrlLength", "CLASS_LABEL") %in% names(values$data_clean))) {
      ggplot(values$data_clean, aes(x = UrlLength, fill = as.factor(CLASS_LABEL))) + 
        geom_density(alpha = 0.6) + 
        labs(title = "Density Plot of URL Length", 
             x = "URL Length", y = "Density",
             fill = "Class") +
        theme_minimal()
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "Required columns not found", cex = 1.5)
    }
  })
  
  # Correlation Plot
  output$correlation_plot <- renderPlot({
    req(values$data_clean)
    
    # Select only numeric columns for correlation
    numeric_data <- values$data_clean[, sapply(values$data_clean, is.numeric)]
    
    if(ncol(numeric_data) >= 2) {
      cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")
      
      # Apply correlation cutoff
      cor_matrix[abs(cor_matrix) < input$corr_cutoff] <- 0
      
      corrplot(cor_matrix, method = "color", type = "upper", 
               tl.cex = 0.7, tl.col = "black", diag = FALSE,
               title = paste("Correlation Matrix (Cutoff:", input$corr_cutoff, ")"))
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "Insufficient numeric columns for correlation", cex = 1.5)
    }
  })
  
  # Split data for training and testing
  observeEvent(input$split_data, {
    req(values$data_clean)
    data <- values$data_clean
    
    if(!"CLASS_LABEL" %in% names(data)) {
      showNotification("CLASS_LABEL column not found in the dataset", type = "error")
      return()
    }
    
    # Convert CLASS_LABEL to factor
    data$CLASS_LABEL <- as.factor(data$CLASS_LABEL)
    
    # Split data
    set.seed(input$random_seed)
    train_index <- sample(1:nrow(data), input$train_test_split/100 * nrow(data))
    values$train_data <- data[train_index, ]
    values$test_data <- data[-train_index, ]
    
    showNotification(paste0("Data split into ", nrow(values$train_data), 
                            " training samples and ", nrow(values$test_data), 
                            " testing samples"), type = "message")
  })
  
  # Train Logistic Regression
  observeEvent(input$train_logistic, {
    req(values$train_data, values$test_data)
    
    withProgress(message = 'Training Logistic Regression...', {
      # Train model
      logistic_model <- glm(CLASS_LABEL ~ ., data = values$train_data, family = "binomial")
      values$logistic_model <- logistic_model
      
      # Predict
      pred_prob <- predict(logistic_model, values$test_data, type = "response")
      pred_class <- ifelse(pred_prob > 0.5, 1, 0)
      
      # Evaluate
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Get other metrics
      if(length(unique(values$test_data$CLASS_LABEL)) == 2) {
        # For binary classification
        roc_obj <- roc(values$test_data$CLASS_LABEL, pred_prob)
        auc_value <- auc(roc_obj)
        
        # Add to metrics table
        new_metrics <- data.frame(
          Model = "Logistic Regression",
          Accuracy = accuracy,
          Sensitivity = sensitivity(as.factor(pred_class), as.factor(values$test_data$CLASS_LABEL), positive = "1"),
          Specificity = specificity(as.factor(pred_class), as.factor(values$test_data$CLASS_LABEL), positive = "1"),
          AUC = auc_value,
          stringsAsFactors = FALSE
        )
        
        # Update metrics table
        values$metrics <- rbind(values$metrics[values$metrics$Model != "Logistic Regression", ], new_metrics)
        
        # Plot ROC curve
        output$logistic_roc <- renderPlot({
          plot(roc_obj, main = "ROC Curve - Logistic Regression", 
               col = "blue", lwd = 2)
          abline(a = 0, b = 1, lty = 2, col = "gray")
          legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), 
                 col = "blue", lwd = 2)
        })
      }
      
      # Display summary
      output$logistic_summary <- renderPrint({
        cat("Logistic Regression Results:\n\n")
        cat("Confusion Matrix:\n")
        print(conf_matrix)
        cat("\nAccuracy:", round(accuracy, 4), "\n")
        if(exists("auc_value")) cat("AUC:", round(auc_value, 4), "\n")
        cat("\nModel Summary:\n")
        print(summary(logistic_model))
      })
    })
  })
  
  # Train KNN
  observeEvent(input$train_knn, {
    req(values$train_data, values$test_data)
    
    withProgress(message = 'Training KNN...', {
      # Prepare data
      train_x <- values$train_data[, !names(values$train_data) %in% "CLASS_LABEL"]
      test_x <- values$test_data[, !names(values$test_data) %in% "CLASS_LABEL"]
      train_y <- values$train_data$CLASS_LABEL
      test_y <- values$test_data$CLASS_LABEL
      
      # Train model and predict
      knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = input$knn_k)
      values$knn_model <- list(k = input$knn_k, train_x = train_x, train_y = train_y)
      
      # Evaluate
      conf_matrix <- table(Predicted = knn_pred, Actual = test_y)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Add to metrics table
      new_metrics <- data.frame(
        Model = "KNN",
        Accuracy = accuracy,
        Sensitivity = sensitivity(knn_pred, test_y, positive = "1"),
        Specificity = specificity(knn_pred, test_y, positive = "1"),
        AUC = NA,  # KNN doesn't directly provide probability estimates for standard ROC curve
        stringsAsFactors = FALSE
      )
      
      # Update metrics table
      values$metrics <- rbind(values$metrics[values$metrics$Model != "KNN", ], new_metrics)
      
      # Plot a visualization (class distribution by features)
      output$knn_plot <- renderPlot({
        if(ncol(train_x) >= 2) {
          # Use first two features to visualize
          plot_data <- data.frame(
            x = test_x[, 1],
            y = test_x[, 2],
            actual = test_y,
            predicted = knn_pred
          )
          
          ggplot(plot_data) +
            geom_point(aes(x = x, y = y, color = predicted, shape = actual), size = 3) +
            labs(title = paste("KNN Classification (k =", input$knn_k, ")"),
                 x = names(test_x)[1], y = names(test_x)[2],
                 color = "Predicted", shape = "Actual") +
            theme_minimal()
        } else {
          plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
          text(1, 1, "Need at least 2 features for visualization", cex = 1.5)
        }
      })
      
      # Display summary
      output$knn_summary <- renderPrint({
        cat("K-Nearest Neighbors Results:\n\n")
        cat("k value:", input$knn_k, "\n\n")
        cat("Confusion Matrix:\n")
        print(conf_matrix)
        cat("\nAccuracy:", round(accuracy, 4), "\n")
        cat("Sensitivity:", round(new_metrics$Sensitivity, 4), "\n")
        cat("Specificity:", round(new_metrics$Specificity, 4), "\n")
      })
    })
  })
  
  # Train Naive Bayes
  observeEvent(input$train_nb, {
    req(values$train_data, values$test_data)
    
    withProgress(message = 'Training Naive Bayes...', {
      # Train model
      nb_model <- naiveBayes(CLASS_LABEL ~ ., data = values$train_data)
      values$nb_model <- nb_model
      
      # Predict
      pred_class <- predict(nb_model, values$test_data)
      pred_prob <- predict(nb_model, values$test_data, type = "raw")
      
      # Evaluate
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Calculate ROC and AUC
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2])
        auc_value <- auc(roc_obj)
        
        # Add to metrics table
        new_metrics <- data.frame(
          Model = "Naive Bayes",
          Accuracy = accuracy,
          Sensitivity = sensitivity(pred_class, values$test_data$CLASS_LABEL, positive = "1"),
          Specificity = specificity(pred_class, values$test_data$CLASS_LABEL, positive = "1"),
          AUC = auc_value,
          stringsAsFactors = FALSE
        )
        
        # Update metrics table
        values$metrics <- rbind(values$metrics[values$metrics$Model != "Naive Bayes", ], new_metrics)
        
        # Plot ROC curve
        output$nb_roc <- renderPlot({
          plot(roc_obj, main = "ROC Curve - Naive Bayes", 
               col = "green", lwd = 2)
          abline(a = 0, b = 1, lty = 2, col = "gray")
          legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), 
                 col = "green", lwd = 2)
        })
      }
      
      # Display summary
      output$nb_summary <- renderPrint({
        cat("Naive Bayes Results:\n\n")
        cat("Confusion Matrix:\n")
        print(conf_matrix)
        cat("\nAccuracy:", round(accuracy, 4), "\n")
        if(exists("auc_value")) cat("AUC:", round(auc_value, 4), "\n")
        
        cat("\nModel Details:\n")
        print(nb_model)
      })
    })
  })
  
  # Train Decision Tree
  observeEvent(input$train_dt, {
    req(values$train_data, values$test_data)
    
    withProgress(message = 'Training Decision Tree...', {
      # Train model
      dt_model <- rpart(CLASS_LABEL ~ ., data = values$train_data, 
                        method = "class", control = rpart.control(maxdepth = input$dt_depth))
      values$dt_model <- dt_model
      
      # Predict
      pred_class <- predict(dt_model, values$test_data, type = "class")
      pred_prob <- predict(dt_model, values$test_data, type = "prob")
      
      # Evaluate
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Calculate ROC and AUC
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2])
        auc_value <- auc(roc_obj)
        
        # Add to metrics table
        new_metrics <- data.frame(
          Model = "Decision Tree",
          Accuracy = accuracy,
          Sensitivity = sensitivity(pred_class, values$test_data$CLASS_LABEL, positive = "1"),
          Specificity = specificity(pred_class, values$test_data$CLASS_LABEL, positive = "1"),
          AUC = auc_value,
          stringsAsFactors = FALSE
        )
        
        # Update metrics table
        values$metrics <- rbind(values$metrics[values$metrics$Model != "Decision Tree", ], new_metrics)
      }
      
      # Plot decision tree
      output$dt_plot <- renderPlot({
        rpart.plot(dt_model, main = "Decision Tree", 
                   extra = 106, box.palette = "RdBu", shadow.col = "gray")
      })
      
      # Display summary
      output$dt_summary <- renderPrint({
        cat("Decision Tree Results:\n\n")
        cat("Confusion Matrix:\n")
        print(conf_matrix)
        cat("\nAccuracy:", round(accuracy, 4), "\n")
        if(exists("auc_value")) cat("AUC:", round(auc_value, 4), "\n")
        
        cat("\nVariable Importance:\n")
        print(dt_model$variable.importance)
      })
    })
  })
  
  # Train Random Forest
  observeEvent(input$train_rf, {
    req(values$train_data, values$test_data)
    
    withProgress(message = 'Training Random Forest...', {
      # Train model
      rf_model <- randomForest(CLASS_LABEL ~ ., data = values$train_data, 
                               ntree = input$rf_trees, importance = TRUE)
      values$rf_model <- rf_model
      
      # Predict
      pred_class <- predict(rf_model, values$test_data)
      pred_prob <- predict(rf_model, values$test_data, type = "prob")
      
      # Evaluate
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Calculate ROC and AUC
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2])
        auc_value <- auc(roc_obj)
        
        # Add to metrics table
        new_metrics <- data.frame(
          Model = "Random Forest",
          Accuracy = accuracy,
          Sensitivity = sensitivity(pred_class, values$test_data$CLASS_LABEL, positive = "1"),
          Specificity = specificity(pred_class, values$test_data$CLASS_LABEL, positive = "1"),
          AUC = auc_value,
          stringsAsFactors = FALSE
        )
        
        # Update metrics table
        values$metrics <- rbind(values$metrics[values$metrics$Model != "Random Forest", ], new_metrics)
      }
      
      # Plot feature importance
      output$rf_importance <- renderPlot({
        importance_df <- as.data.frame(importance(rf_model))
        importance_df$Features <- rownames(importance_df)
        
        ggplot(importance_df, aes(x = reorder(Features, MeanDecreaseGini), y = MeanDecreaseGini)) +
          geom_bar(stat = "identity", fill = "#4daf4a") +
          coord_flip() +
          labs(title = "Random Forest - Feature Importance",
               x = "Features",
               y = "Mean Decrease in Gini Index") +
          theme_minimal()
      })
      
      # Display summary
      output$rf_summary <- renderPrint({
        cat("Random Forest Results:\n\n")
        cat("Confusion Matrix:\n")
        print(conf_matrix)
        cat("\nAccuracy:", round(accuracy, 4), "\n")
        if(exists("auc_value")) cat("AUC:", round(auc_value, 4), "\n")
        
        cat("\nModel Details:\n")
        print(rf_model)
      })
    })
  })
  
  # Model Comparison - Accuracy Plot
  output$accuracy_plot <- renderPlot({
    req(nrow(values$metrics) > 0)
    
    ggplot(values$metrics, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")), 
                vjust = -0.3, size = 5) +
      labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # Model Comparison - ROC Curves
  output$roc_comparison <- renderPlot({
    req(values$test_data)
    
    # Create an empty plot
    plot(x = c(0, 1), y = c(0, 1), type = "n", 
         xlim = c(0, 1), ylim = c(0, 1),
         xlab = "False Positive Rate", ylab = "True Positive Rate",
         main = "ROC Curve Comparison")
    abline(a = 0, b = 1, lty = 2, col = "gray")
    
    # Add ROC curves for each model
    colors <- c("blue", "green", "red", "purple", "orange")
    legend_text <- c()
    
    # Check if models exist and add their ROC curves
    i <- 1
    
    # Logistic Regression
    if(!is.null(values$logistic_model)) {
      pred_prob <- predict(values$logistic_model, values$test_data, type = "response")
      roc_obj <- roc(values$test_data$CLASS_LABEL, pred_prob, plot = FALSE)
      plot(roc_obj, add = TRUE, col = colors[i], lwd = 2)
      legend_text <- c(legend_text, paste0("Logistic Regression (AUC = ", round(auc(roc_obj), 3), ")"))
      i <- i + 1
    }
    
    # Naive Bayes
    if(!is.null(values$nb_model)) {
      pred_prob <- predict(values$nb_model, values$test_data, type = "raw")
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2], plot = FALSE)
        plot(roc_obj, add = TRUE, col = colors[i], lwd = 2)
        legend_text <- c(legend_text, paste0("Naive Bayes (AUC = ", round(auc(roc_obj), 3), ")"))
        i <- i + 1
      }
    }
    
    # Decision Tree
    if(!is.null(values$dt_model)) {
      pred_prob <- predict(values$dt_model, values$test_data, type = "prob")
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2], plot = FALSE)
        plot(roc_obj, add = TRUE, col = colors[i], lwd = 2)
        legend_text <- c(legend_text, paste0("Decision Tree (AUC = ", round(auc(roc_obj), 3), ")"))
        i <- i + 1
      }
    }
    
    # Random Forest
    if(!is.null(values$rf_model)) {
      pred_prob <- predict(values$rf_model, values$test_data, type = "prob")
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2], plot = FALSE)
        plot(roc_obj, add = TRUE, col = colors[i], lwd = 2)
        legend_text <- c(legend_text, paste0("Random Forest (AUC = ", round(auc(roc_obj), 3), ")"))
        i <- i + 1
      }
    }
    
    # Add legend if any models were plotted
    if(length(legend_text) > 0) {
      legend("bottomright", legend = legend_text, col = colors[1:length(legend_text)], 
             lwd = 2, cex = 0.8)
    }
  })
  
  # Metrics Table
  output$metrics_table <- DT::renderDataTable({
    req(values$metrics)
    
    # Format metrics for display
    display_metrics <- values$metrics
    display_metrics$Accuracy <- paste0(round(display_metrics$Accuracy * 100, 2), "%")
    display_metrics$Sensitivity <- paste0(round(display_metrics$Sensitivity * 100, 2), "%")
    display_metrics$Specificity <- paste0(round(display_metrics$Specificity * 100, 2), "%")
    display_metrics$AUC <- round(display_metrics$AUC, 3)
    
    DT::datatable(display_metrics, 
                  options = list(dom = 't', 
                                 scrollX = TRUE,
                                 pageLength = nrow(display_metrics)))
  })
  
  # Confusion Matrix Visualization
  output$confusion_matrix <- renderPlot({
    req(values$test_data, input$cm_model)
    
    # Get the selected model and predictions
    if(input$cm_model == "Logistic Regression" && !is.null(values$logistic_model)) {
      pred_prob <- predict(values$logistic_model, values$test_data, type = "response")
      pred_class <- ifelse(pred_prob > 0.5, 1, 0)
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
    } else if(input$cm_model == "KNN" && !is.null(values$knn_model)) {
      knn_pred <- knn(train = values$knn_model$train_x, 
                      test = values$test_data[, !names(values$test_data) %in% "CLASS_LABEL"], 
                      cl = values$knn_model$train_y, 
                      k = values$knn_model$k)
      conf_matrix <- table(Predicted = knn_pred, Actual = values$test_data$CLASS_LABEL)
    } else if(input$cm_model == "Naive Bayes" && !is.null(values$nb_model)) {
      pred_class <- predict(values$nb_model, values$test_data)
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
    } else if(input$cm_model == "Decision Tree" && !is.null(values$dt_model)) {
      pred_class <- predict(values$dt_model, values$test_data, type = "class")
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
    } else if(input$cm_model == "Random Forest" && !is.null(values$rf_model)) {
      pred_class <- predict(values$rf_model, values$test_data)
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
    } else {
      # No valid model selected
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "Selected model not trained yet", cex = 1.5)
      return()
    }
    
    # Visualize confusion matrix
    conf_matrix_df <- as.data.frame(as.table(conf_matrix))
    colnames(conf_matrix_df) <- c("Predicted", "Actual", "Freq")
    
    # Calculate total for each actual class (for percentages)
    totals <- aggregate(Freq ~ Actual, data = conf_matrix_df, sum)
    conf_matrix_df <- merge(conf_matrix_df, totals, by = "Actual", suffixes = c("", "_total"))
    conf_matrix_df$Percentage <- conf_matrix_df$Freq / conf_matrix_df$Freq_total * 100
    
    # Create the heatmap
    ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = paste0(Freq, "\n(", round(Percentage, 1), "%)")), 
                color = "black", size = 4) +
      scale_fill_gradient(low = "white", high = "steelblue") +
      labs(title = paste("Confusion Matrix -", input$cm_model),
           x = "Predicted Class",
           y = "Actual Class") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  })
}

# Run the application
shinyApp(ui, server)
