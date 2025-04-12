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
  
  # Load data when file is uploaded
  observeEvent(input$file, {
    req(input$file)
    values$data <- read.csv(input$file$datapath, 
                            header = input$header,
                            sep = input$sep,
                            na.strings = c("", "NA"))
  })
  
  # Sample data for demonstration
  observeEvent(input$use_sample, {
    values$data <- read.csv("Phishing_Legitimate.csv", na.strings = c("", "NA"))
    # If the sample file doesn't work, create mock data
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
    
    # Calculate missing values
    missing_values <- colSums(is.na(values$data))
    missing_percentage <- (missing_values / nrow(values$data)) * 100
    
    if(sum(missing_values) > 0) {
      # Create a barplot of missing values percentages
      barplot(missing_percentage, 
              main = "Missing Values by Column", 
              xlab = "Column", 
              ylab = "Percentage Missing",
              las = 2,
              col = "steelblue")
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
          # Make sure mice package is loaded
          if (!require("mice")) install.packages("mice")
          library(mice)
          
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
      # Ensure ggplot2 is loaded
      if (!require("ggplot2")) install.packages("ggplot2")
      library(ggplot2)
      
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
      # Ensure ggplot2 is loaded
      if (!require("ggplot2")) install.packages("ggplot2")
      library(ggplot2)
      
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
      # Ensure ggplot2 is loaded
      if (!require("ggplot2")) install.packages("ggplot2")
      library(ggplot2)
      
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
      # Ensure ggplot2 is loaded
      if (!require("ggplot2")) install.packages("ggplot2")
      library(ggplot2)
      
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
    
    # Ensure corrplot is loaded
    if (!require("corrplot")) install.packages("corrplot")
    library(corrplot)
    
    # Select only numeric columns for correlation
    numeric_data <- values$data_clean[, sapply(values$data_clean, is.numeric)]
    
    if(ncol(numeric_data) >= 2) {
      cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")
      
      # Apply correlation cutoff if specified
      if (!is.null(input$corr_cutoff)) {
        cor_matrix[abs(cor_matrix) < input$corr_cutoff] <- 0
      }
      
      corrplot(cor_matrix, method = "color", type = "upper", 
               tl.cex = 0.7, tl.col = "black", diag = FALSE,
               title = paste("Correlation Matrix"))
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
    # Using 80% for training as in the first file
    train_size <- 0.8
    if (!is.null(input$train_test_split)) {
      train_size <- input$train_test_split/100
    }
    
    train_index <- sample(1:nrow(data), train_size * nrow(data))
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
      # Train model using the same approach as in the first file
      logistic_model <- glm(CLASS_LABEL ~ ., data = values$train_data, family = "binomial")
      values$logistic_model <- logistic_model
      
      # Predict
      pred_prob <- predict(logistic_model, values$test_data, type = "response")
      pred_class <- ifelse(pred_prob > 0.5, 1, 0)
      
      # Evaluate
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Get other metrics
      # Make sure pROC is loaded
      if (!require("pROC")) install.packages("pROC")
      library(pROC)
      
      if(length(unique(values$test_data$CLASS_LABEL)) == 2) {
        # For binary classification
        roc_obj <- roc(values$test_data$CLASS_LABEL, pred_prob)
        auc_value <- auc(roc_obj)
        
        # Use caret for additional metrics
        if (!require("caret")) install.packages("caret")
        library(caret)
        
        # Convert to factors for caret metrics
        pred_class_factor <- factor(pred_class, levels = levels(values$test_data$CLASS_LABEL))
        
        # Calculate sensitivity and specificity
        cm <- confusionMatrix(pred_class_factor, values$test_data$CLASS_LABEL)
        sensitivity_val <- cm$byClass["Sensitivity"]
        specificity_val <- cm$byClass["Specificity"]
        
        # Add to metrics table
        new_metrics <- data.frame(
          Model = "Logistic Regression",
          Accuracy = accuracy,
          Sensitivity = sensitivity_val,
          Specificity = specificity_val,
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
    
    # Make sure class package is loaded
    if (!require("class")) install.packages("class")
    library(class)
    
    withProgress(message = 'Training KNN...', {
      # Prepare data
      train_x <- values$train_data[, !names(values$train_data) %in% "CLASS_LABEL"]
      test_x <- values$test_data[, !names(values$test_data) %in% "CLASS_LABEL"]
      train_y <- values$train_data$CLASS_LABEL
      test_y <- values$test_data$CLASS_LABEL
      
      # Train model and predict
      k_value <- 5
      if (!is.null(input$knn_k)) {
        k_value <- input$knn_k
      }
      
      knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = k_value)
      values$knn_model <- list(k = k_value, train_x = train_x, train_y = train_y)
      
      # Evaluate
      conf_matrix <- table(Predicted = knn_pred, Actual = test_y)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Use caret for additional metrics
      if (!require("caret")) install.packages("caret")
      library(caret)
      
      # Calculate sensitivity and specificity
      cm <- confusionMatrix(knn_pred, test_y)
      sensitivity_val <- cm$byClass["Sensitivity"]
      specificity_val <- cm$byClass["Specificity"]
      
      # Add to metrics table
      new_metrics <- data.frame(
        Model = "KNN",
        Accuracy = accuracy,
        Sensitivity = sensitivity_val,
        Specificity = specificity_val,
        AUC = NA,  # KNN doesn't directly provide probability estimates for standard ROC curve
        stringsAsFactors = FALSE
      )
      
      # Update metrics table
      values$metrics <- rbind(values$metrics[values$metrics$Model != "KNN", ], new_metrics)
      
      # Ensure ggplot2 is loaded
      if (!require("ggplot2")) install.packages("ggplot2")
      library(ggplot2)
      
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
            labs(title = paste("KNN Classification (k =", k_value, ")"),
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
        cat("k value:", k_value, "\n\n")
        cat("Confusion Matrix:\n")
        print(conf_matrix)
        cat("\nAccuracy:", round(accuracy, 4), "\n")
        cat("Sensitivity:", round(sensitivity_val, 4), "\n")
        cat("Specificity:", round(specificity_val, 4), "\n")
      })
    })
  })
  
  # Train Naive Bayes
  observeEvent(input$train_nb, {
    req(values$train_data, values$test_data)
    
    # Make sure e1071 package is loaded
    if (!require("e1071")) install.packages("e1071")
    library(e1071)
    
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
      
      # Make sure pROC is loaded
      if (!require("pROC")) install.packages("pROC")
      library(pROC)
      
      # Calculate ROC and AUC
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2])
        auc_value <- auc(roc_obj)
        
        # Use caret for additional metrics
        if (!require("caret")) install.packages("caret")
        library(caret)
        
        # Calculate sensitivity and specificity
        cm <- confusionMatrix(pred_class, values$test_data$CLASS_LABEL)
        sensitivity_val <- cm$byClass["Sensitivity"]
        specificity_val <- cm$byClass["Specificity"]
        
        # Add to metrics table
        new_metrics <- data.frame(
          Model = "Naive Bayes",
          Accuracy = accuracy,
          Sensitivity = sensitivity_val,
          Specificity = specificity_val,
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
    
    # Make sure rpart and rpart.plot are loaded
    if (!require("rpart")) install.packages("rpart")
    if (!require("rpart.plot")) install.packages("rpart.plot")
    library(rpart)
    library(rpart.plot)
    
    withProgress(message = 'Training Decision Tree...', {
      # Default max depth
      max_depth <- 6
      if (!is.null(input$dt_depth)) {
        max_depth <- input$dt_depth
      }
      
      # Train model
      dt_model <- rpart(CLASS_LABEL ~ ., data = values$train_data, 
                        method = "class", control = rpart.control(maxdepth = max_depth))
      values$dt_model <- dt_model
      
      # Predict
      pred_class <- predict(dt_model, values$test_data, type = "class")
      pred_prob <- predict(dt_model, values$test_data, type = "prob")
      
      # Evaluate
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Make sure pROC is loaded
      if (!require("pROC")) install.packages("pROC")
      library(pROC)
      
      # Calculate ROC and AUC
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2])
        auc_value <- auc(roc_obj)
        
        # Use caret for additional metrics
        if (!require("caret")) install.packages("caret")
        library(caret)
        
        # Calculate sensitivity and specificity
        cm <- confusionMatrix(pred_class, values$test_data$CLASS_LABEL)
        sensitivity_val <- cm$byClass["Sensitivity"]
        specificity_val <- cm$byClass["Specificity"]
        
        # Add to metrics table
        new_metrics <- data.frame(
          Model = "Decision Tree",
          Accuracy = accuracy,
          Sensitivity = sensitivity_val,
          Specificity = specificity_val,
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
        cat("Sensitivity:", round(sensitivity_val, 4), "\n")
        cat("Specificity:", round(specificity_val, 4), "\n")
        # Variable importance plot
        output$dt_var_imp <- renderPlot({
          # Plot variable importance
          imp <- dt_model$variable.importance
          if(length(imp) > 0) {
            # Sort variable importance in descending order
            imp_sorted <- sort(imp, decreasing = TRUE)
            # Plot as a bar chart
            barplot(imp_sorted, main = "Variable Importance - Decision Tree",
                    col = "skyblue", horiz = TRUE, las = 1)
          } else {
            plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
            text(1, 1, "No variable importance available", cex = 1.5)
          }
        })
      })
    })
  })
  
  # Train Random Forest
  observeEvent(input$train_rf, {
    req(values$train_data, values$test_data)
    
    # Make sure randomForest package is loaded
    if (!require("randomForest")) install.packages("randomForest")
    library(randomForest)
    
    withProgress(message = 'Training Random Forest...', {
      # Default number of trees
      n_trees <- 100
      if (!is.null(input$rf_trees)) {
        n_trees <- input$rf_trees
      }
      
      # Train model
      set.seed(input$random_seed)
      rf_model <- randomForest(CLASS_LABEL ~ ., data = values$train_data, 
                               ntree = n_trees, importance = TRUE)
      values$rf_model <- rf_model
      
      # Predict
      pred_class <- predict(rf_model, values$test_data)
      pred_prob <- predict(rf_model, values$test_data, type = "prob")
      
      # Evaluate
      conf_matrix <- table(Predicted = pred_class, Actual = values$test_data$CLASS_LABEL)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Make sure pROC is loaded
      if (!require("pROC")) install.packages("pROC")
      library(pROC)
      
      # Calculate ROC and AUC
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(as.numeric(values$test_data$CLASS_LABEL) - 1, pred_prob[, 2])
        auc_value <- auc(roc_obj)
        
        # Use caret for additional metrics
        if (!require("caret")) install.packages("caret")
        library(caret)
        
        # Calculate sensitivity and specificity
        cm <- confusionMatrix(pred_class, values$test_data$CLASS_LABEL)
        sensitivity_val <- cm$byClass["Sensitivity"]
        specificity_val <- cm$byClass["Specificity"]
        
        # Add to metrics table
        new_metrics <- data.frame(
          Model = "Random Forest",
          Accuracy = accuracy,
          Sensitivity = sensitivity_val,
          Specificity = specificity_val,
          AUC = auc_value,
          stringsAsFactors = FALSE
        )
        
        # Update metrics table
        values$metrics <- rbind(values$metrics[values$metrics$Model != "Random Forest", ], new_metrics)
        
        # Plot ROC curve
        output$rf_roc <- renderPlot({
          plot(roc_obj, main = "ROC Curve - Random Forest", 
               col = "purple", lwd = 2)
          abline(a = 0, b = 1, lty = 2, col = "gray")
          legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), 
                 col = "purple", lwd = 2)
        })
      }
      
      # Variable importance plot
      output$rf_var_imp <- renderPlot({
        # Plot variable importance
        if (!is.null(rf_model$importance)) {
          # Ensure ggplot2 is loaded
          if (!require("ggplot2")) install.packages("ggplot2")
          library(ggplot2)
          
          # Get mean decrease in accuracy
          imp_df <- as.data.frame(importance(rf_model))
          imp_df$Variable <- rownames(imp_df)
          
          # Sort by mean decrease in accuracy
          imp_df <- imp_df[order(imp_df$MeanDecreaseAccuracy, decreasing = TRUE), ]
          
          # Plot top 15 most important variables
          imp_df <- head(imp_df, 15)
          
          ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
            geom_bar(stat = "identity", fill = "darkgreen") +
            coord_flip() +
            labs(title = "Variable Importance - Random Forest",
                 x = "", y = "Mean Decrease in Accuracy") +
            theme_minimal()
        } else {
          plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
          text(1, 1, "No variable importance available", cex = 1.5)
        }
      })
      
      # Plot error rate
      output$rf_error_plot <- renderPlot({
        plot(rf_model, main = "Random Forest Error Rate vs Number of Trees")
      })
      
      # Display summary
      output$rf_summary <- renderPrint({
        cat("Random Forest Results:\n\n")
        cat("Number of trees:", n_trees, "\n\n")
        cat("Confusion Matrix:\n")
        print(conf_matrix)
        cat("\nAccuracy:", round(accuracy, 4), "\n")
        if(exists("auc_value")) cat("AUC:", round(auc_value, 4), "\n")
        cat("Sensitivity:", round(sensitivity_val, 4), "\n")
        cat("Specificity:", round(specificity_val, 4), "\n")
        
        cat("\nModel Details:\n")
        print(rf_model)
      })
    })
  })
  
  # Metrics Comparison Plot
  output$metrics_comparison <- renderPlot({
    req(values$metrics)
    if(nrow(values$metrics) == 0) {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, "No model metrics available. Train models first.", cex = 1.5)
      return()
    }
    
    # Ensure ggplot2 is loaded
    if (!require("ggplot2")) install.packages("ggplot2")
    library(ggplot2)
    
    # Reshape data for plotting
    metrics_long <- reshape2::melt(values$metrics, id.vars = "Model")
    
    # Plot metrics
    ggplot(metrics_long, aes(x = Model, y = value, fill = variable)) +
      geom_bar(stat = "identity", position = "dodge") +
      labs(title = "Model Performance Comparison",
           x = "Model", y = "Value", fill = "Metric") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ylim(0, 1)
  })
  
  # Metrics Comparison Table
  output$metrics_table <- DT::renderDataTable({
    req(values$metrics)
    DT::datatable(values$metrics, 
                  options = list(pageLength = 10, searching = FALSE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("Accuracy", "Sensitivity", "Specificity", "AUC"), digits = 4)
  })
  
  # Make predictions on new data
  observeEvent(input$predict_btn, {
    req(input$predict_input)
    
    # Try to parse the input JSON
    tryCatch({
      new_data <- jsonlite::fromJSON(input$predict_input)
      
      # Convert to data frame if it's a list
      if(is.list(new_data) && !is.data.frame(new_data)) {
        new_data <- as.data.frame(new_data)
      }
      
      # Check available models
      available_models <- c()
      if(!is.null(values$logistic_model)) available_models <- c(available_models, "Logistic Regression")
      if(!is.null(values$knn_model)) available_models <- c(available_models, "KNN")
      if(!is.null(values$nb_model)) available_models <- c(available_models, "Naive Bayes")
      if(!is.null(values$dt_model)) available_models <- c(available_models, "Decision Tree")
      if(!is.null(values$rf_model)) available_models <- c(available_models, "Random Forest")
      
      if(length(available_models) == 0) {
        output$predict_result <- renderText("No models available. Please train models first.")
        return()
      }
      
      # Make predictions with all available models
      results <- data.frame(Model = character(), Prediction = character(), stringsAsFactors = FALSE)
      
      # Logistic Regression
      if(!is.null(values$logistic_model)) {
        pred_prob <- predict(values$logistic_model, new_data, type = "response")
        pred_class <- ifelse(pred_prob > 0.5, "Phishing (1)", "Legitimate (0)")
        results <- rbind(results, data.frame(Model = "Logistic Regression", 
                                             Prediction = pred_class,
                                             Probability = round(pred_prob, 4),
                                             stringsAsFactors = FALSE))
      }
      
      # KNN
      if(!is.null(values$knn_model)) {
        # Make sure class columns match
        common_cols <- intersect(names(new_data), names(values$knn_model$train_x))
        if(length(common_cols) > 0) {
          knn_pred <- knn(train = values$knn_model$train_x[, common_cols], 
                          test = new_data[, common_cols], 
                          cl = values$knn_model$train_y, 
                          k = values$knn_model$k)
          pred_class <- ifelse(knn_pred == 1, "Phishing (1)", "Legitimate (0)")
          results <- rbind(results, data.frame(Model = "KNN", 
                                               Prediction = pred_class,
                                               Probability = NA,
                                               stringsAsFactors = FALSE))
        } else {
          results <- rbind(results, data.frame(Model = "KNN", 
                                               Prediction = "Feature mismatch error",
                                               Probability = NA,
                                               stringsAsFactors = FALSE))
        }
      }
      
      # Naive Bayes
      if(!is.null(values$nb_model)) {
        tryCatch({
          pred_class <- predict(values$nb_model, new_data)
          pred_prob <- predict(values$nb_model, new_data, type = "raw")
          pred_class_label <- ifelse(pred_class == 1, "Phishing (1)", "Legitimate (0)")
          results <- rbind(results, data.frame(Model = "Naive Bayes", 
                                               Prediction = pred_class_label,
                                               Probability = round(pred_prob[, 2], 4),
                                               stringsAsFactors = FALSE))
        }, error = function(e) {
          results <- rbind(results, data.frame(Model = "Naive Bayes", 
                                               Prediction = paste("Error:", e$message),
                                               Probability = NA,
                                               stringsAsFactors = FALSE))
        })
      }
      
      # Decision Tree
      if(!is.null(values$dt_model)) {
        tryCatch({
          pred_class <- predict(values$dt_model, new_data, type = "class")
          pred_prob <- predict(values$dt_model, new_data, type = "prob")
          pred_class_label <- ifelse(pred_class == 1, "Phishing (1)", "Legitimate (0)")
          results <- rbind(results, data.frame(Model = "Decision Tree", 
                                               Prediction = pred_class_label,
                                               Probability = round(pred_prob[, 2], 4),
                                               stringsAsFactors = FALSE))
        }, error = function(e) {
          results <- rbind(results, data.frame(Model = "Decision Tree", 
                                               Prediction = paste("Error:", e$message),
                                               Probability = NA,
                                               stringsAsFactors = FALSE))
        })
      }
      
      # Random Forest
      if(!is.null(values$rf_model)) {
        tryCatch({
          pred_class <- predict(values$rf_model, new_data)
          pred_prob <- predict(values$rf_model, new_data, type = "prob")
          pred_class_label <- ifelse(pred_class == 1, "Phishing (1)", "Legitimate (0)")
          results <- rbind(results, data.frame(Model = "Random Forest", 
                                               Prediction = pred_class_label,
                                               Probability = round(pred_prob[, 2], 4),
                                               stringsAsFactors = FALSE))
        }, error = function(e) {
          results <- rbind(results, data.frame(Model = "Random Forest", 
                                               Prediction = paste("Error:", e$message),
                                               Probability = NA,
                                               stringsAsFactors = FALSE))
        })
      }
      
      # Display results
      output$predict_table <- DT::renderDataTable({
        DT::datatable(results, options = list(pageLength = 10), rownames = FALSE)
      })
      
      # Overall verdict based on majority vote
      phishing_count <- sum(results$Prediction == "Phishing (1)", na.rm = TRUE)
      legitimate_count <- sum(results$Prediction == "Legitimate (0)", na.rm = TRUE)
      
      verdict <- ifelse(phishing_count > legitimate_count, 
                        "Majority verdict: Phishing Website", 
                        "Majority verdict: Legitimate Website")
      
      if(phishing_count == legitimate_count) {
        verdict <- "Verdict: Inconclusive (equal votes)"
      }
      
      output$predict_result <- renderText({
        paste0("Prediction Results:\n\n", verdict, "\n\nPhishing votes: ", 
               phishing_count, ", Legitimate votes: ", legitimate_count)
      })
      
    }, error = function(e) {
      output$predict_result <- renderText(paste("Error in input data:", e$message,
                                                "\n\nPlease provide a valid JSON format."))
    })
  })
  
  # Example Input Button
  observeEvent(input$example_input, {
    example_data <- data.frame(
      UrlLength = 75,
      NumDots = 3,
      NumDash = 1,
      NumUnderscore = 0,
      NumPercent = 0,
      NumAmpersand = 0,
      NumSlash = 5,
      HasHttps = 0,
      HasIP = 0,
      DomainLength = 12,
      NumSubdomains = 2,
      PathLength = 35,
      QueryLength = 25,
      NumParameters = 3
    )
    
    # Convert to JSON
    json_data <- jsonlite::toJSON(example_data, pretty = TRUE)
    
    # Update the textInput
    updateTextAreaInput(session, "predict_input", value = json_data)
  })
  
  # Download Model Report
  output$download_report <- downloadHandler(
    filename = function() {
      paste("phishing-detection-report-", Sys.Date(), ".html", sep = "")
    },
    content = function(file) {
      # Make sure rmarkdown is installed
      if (!require("rmarkdown")) install.packages("rmarkdown")
      
      # Create a temporary Rmd file
      temp_report <- file.path(tempdir(), "report.Rmd")
      
      # Add models to the report environment
      report_env <- new.env()
      report_env$metrics <- values$metrics
      report_env$logistic_model <- values$logistic_model
      report_env$knn_model <- values$knn_model
      report_env$nb_model <- values$nb_model
      report_env$dt_model <- values$dt_model
      report_env$rf_model <- values$rf_model
      report_env$test_data <- values$test_data
      
      # Write the Rmd content
      cat("---
title: \"Phishing Website Detection Report\"
date: \"`r Sys.Date()`\"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)
library(ggplot2)
library(knitr)
library(dplyr)
```

## Model Performance Summary

```{r}
kable(metrics, caption = \"Model Performance Metrics\")
```

```{r, fig.width=10, fig.height=6}
# Plot metrics comparison
if(nrow(metrics) > 0) {
  metrics_long <- reshape2::melt(metrics, id.vars = \"Model\")
  
  ggplot(metrics_long, aes(x = Model, y = value, fill = variable)) +
    geom_bar(stat = \"identity\", position = \"dodge\") +
    labs(title = \"Model Performance Comparison\",
         x = \"Model\", y = \"Value\", fill = \"Metric\") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(0, 1)
}
```

## Model Details

", file = temp_report)
      
      # Add Logistic Regression details if available
      if(!is.null(values$logistic_model)) {
        cat("
### Logistic Regression

```{r}
if(!is.null(logistic_model)) {
  # Summary
  cat(\"Model Summary:\\n\")
  print(summary(logistic_model))
  
  # Make predictions on test data
  pred_prob <- predict(logistic_model, test_data, type = \"response\")
  pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  
  # Confusion matrix
  conf_matrix <- table(Predicted = pred_class, Actual = test_data$CLASS_LABEL)
  cat(\"\\nConfusion Matrix:\\n\")
  print(conf_matrix)
  
  # ROC curve
  if(require(pROC)) {
    roc_obj <- roc(test_data$CLASS_LABEL, pred_prob)
    plot(roc_obj, main = \"ROC Curve - Logistic Regression\", 
        col = \"blue\", lwd = 2)
    abline(a = 0, b = 1, lty = 2, col = \"gray\")
    legend(\"bottomright\", legend = paste(\"AUC =\", round(auc(roc_obj), 3)), 
          col = \"blue\", lwd = 2)
  }
}
```
", file = temp_report, append = TRUE)
      }

# Add Decision Tree details if available
if(!is.null(values$dt_model)) {
  cat("
### Decision Tree

```{r, fig.width=10, fig.height=8}
if(!is.null(dt_model) && require(rpart.plot)) {
  # Plot tree
  rpart.plot(dt_model, main = \"Decision Tree\", 
             extra = 106, box.palette = \"RdBu\")
  
  # Variable importance
  imp <- dt_model$variable.importance
  if(length(imp) > 0) {
    # Sort variable importance in descending order
    imp_sorted <- sort(imp, decreasing = TRUE)
    # Plot as a bar chart
    barplot(imp_sorted, main = \"Variable Importance - Decision Tree\",
            col = \"skyblue\", horiz = TRUE, las = 1)
  }
  
  # Confusion matrix
  pred_class <- predict(dt_model, test_data, type = \"class\")
  conf_matrix <- table(Predicted = pred_class, Actual = test_data$CLASS_LABEL)
  cat(\"\\nConfusion Matrix:\\n\")
  print(conf_matrix)
}
```
", file = temp_report, append = TRUE)
}

# Add Random Forest details if available
if(!is.null(values$rf_model)) {
  cat("
### Random Forest

```{r, fig.width=10, fig.height=6}
if(!is.null(rf_model)) {
  # Plot error rate
  plot(rf_model, main = \"Random Forest Error Rate vs Number of Trees\")
  
  # Variable importance
  if(!is.null(rf_model$importance)) {
    # Get mean decrease in accuracy
    imp_df <- as.data.frame(importance(rf_model))
    imp_df$Variable <- rownames(imp_df)
    
    # Sort by mean decrease in accuracy
    imp_df <- imp_df[order(imp_df$MeanDecreaseAccuracy, decreasing = TRUE), ]
    
    # Plot top 15 most important variables
    imp_df <- head(imp_df, 15)
    
    ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
      geom_bar(stat = \"identity\", fill = \"darkgreen\") +
      coord_flip() +
      labs(title = \"Variable Importance - Random Forest\",
           x = \"\", y = \"Mean Decrease in Accuracy\") +
      theme_minimal()
  }
  
  # Confusion matrix
  pred_class <- predict(rf_model, test_data)
  conf_matrix <- table(Predicted = pred_class, Actual = test_data$CLASS_LABEL)
  cat(\"\\nConfusion Matrix:\\n\")
  print(conf_matrix)
}
```
", file = temp_report, append = TRUE)
}

# Render the report
rmarkdown::render(temp_report, output_file = file, 
                  envir = report_env, 
                  quiet = TRUE)
    }
  )
  
  # Feature Recommendations
  output$feature_recommendations <- renderPrint({
    req(values$metrics)
    
    if(nrow(values$metrics) == 0) {
      cat("Train models first to get feature recommendations.")
      return()
    }
    
    cat("Feature Importance Analysis:\n\n")
    
    # Check if Random Forest model is available (best for feature importance)
    if(!is.null(values$rf_model)) {
      cat("Based on Random Forest Variable Importance:\n")
      
      # Get importance measures
      imp <- importance(values$rf_model)
      imp_df <- data.frame(
        Feature = rownames(imp),
        Importance = imp[, "MeanDecreaseAccuracy"],
        stringsAsFactors = FALSE
      )
      
      # Sort by importance
      imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
      
      # Display top features
      cat("\nTop 5 Most Important Features:\n")
      for(i in 1:min(5, nrow(imp_df))) {
        cat(paste0(i, ". ", imp_df$Feature[i], " (Importance: ", 
                   round(imp_df$Importance[i], 4), ")\n"))
      }
      
      # Recommendations based on importance
      cat("\nRecommendations:\n")
      cat("- Focus on these top features for phishing detection\n")
      cat("- Consider creating interaction terms with these features\n")
      
      # Check best performing model
      best_model <- values$metrics[which.max(values$metrics$Accuracy), ]
      cat("\nBest Performing Model: ", best_model$Model, 
          " (Accuracy: ", round(best_model$Accuracy, 4), ")\n", sep = "")
      
    } else if(!is.null(values$dt_model)) {
      # If Random Forest not available, use Decision Tree
      cat("Based on Decision Tree Variable Importance:\n")
      
      # Get importance measures
      imp <- values$dt_model$variable.importance
      imp_df <- data.frame(
        Feature = names(imp),
        Importance = imp,
        stringsAsFactors = FALSE
      )
      
      # Sort by importance
      imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
      
      # Display top features
      cat("\nTop 5 Most Important Features:\n")
      for(i in 1:min(5, nrow(imp_df))) {
        cat(paste0(i, ". ", imp_df$Feature[i], " (Importance: ", 
                   round(imp_df$Importance[i], 4), ")\n"))
      }
      
      # Recommendations based on importance
      cat("\nRecommendations:\n")
      cat("- Focus on these top features for phishing detection\n")
      cat("- Consider feature engineering to enhance these attributes\n")
    } else {
      cat("Train Decision Tree or Random Forest models to get feature importance analysis.")
    }
  })
  
  # Model Ensemble
  observeEvent(input$create_ensemble, {
    req(values$test_data)
    
    # Check if at least 2 models are available
    available_models <- c()
    if(!is.null(values$logistic_model)) available_models <- c(available_models, "logistic")
    if(!is.null(values$knn_model)) available_models <- c(available_models, "knn")
    if(!is.null(values$nb_model)) available_models <- c(available_models, "nb")
    if(!is.null(values$dt_model)) available_models <- c(available_models, "dt")
    if(!is.null(values$rf_model)) available_models <- c(available_models, "rf")
    
    if(length(available_models) < 2) {
      output$ensemble_result <- renderText("At least 2 trained models are required for ensemble.")
      return()
    }
    
    withProgress(message = 'Creating Ensemble Model...', {
      # Get predictions from all available models
      predictions <- list()
      
      # Logistic Regression
      if("logistic" %in% available_models) {
        pred_prob <- predict(values$logistic_model, values$test_data, type = "response")
        predictions$logistic <- ifelse(pred_prob > 0.5, 1, 0)
      }
      
      # KNN
      if("knn" %in% available_models) {
        # Prepare data
        train_x <- values$train_data[, !names(values$train_data) %in% "CLASS_LABEL"]
        test_x <- values$test_data[, !names(values$test_data) %in% "CLASS_LABEL"]
        train_y <- values$train_data$CLASS_LABEL
        
        knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = values$knn_model$k)
        predictions$knn <- as.numeric(as.character(knn_pred))
      }
      
      # Naive Bayes
      if("nb" %in% available_models) {
        pred_class <- predict(values$nb_model, values$test_data)
        predictions$nb <- as.numeric(as.character(pred_class))
      }
      
      # Decision Tree
      if("dt" %in% available_models) {
        pred_class <- predict(values$dt_model, values$test_data, type = "class")
        predictions$dt <- as.numeric(as.character(pred_class))
      }
      
      # Random Forest
      if("rf" %in% available_models) {
        pred_class <- predict(values$rf_model, values$test_data)
        predictions$rf <- as.numeric(as.character(pred_class))
      }
      
      # Combine predictions (majority vote)
      pred_matrix <- do.call(cbind, predictions)
      ensemble_pred <- apply(pred_matrix, 1, function(x) {
        if(mean(x) > 0.5) 1 else 0
      })
      
      # Evaluate ensemble
      conf_matrix <- table(Predicted = ensemble_pred, Actual = values$test_data$CLASS_LABEL)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      # Use caret for additional metrics
      if (!require("caret")) install.packages("caret")
      library(caret)
      
      # Convert to factors for caret metrics
      pred_class_factor <- factor(ensemble_pred, levels = levels(values$test_data$CLASS_LABEL))
      
      # Calculate sensitivity and specificity
      cm <- confusionMatrix(pred_class_factor, values$test_data$CLASS_LABEL)
      sensitivity_val <- cm$byClass["Sensitivity"]
      specificity_val <- cm$byClass["Specificity"]
      
      # Add to metrics table
      new_metrics <- data.frame(
        Model = "Ensemble (Majority Vote)",
        Accuracy = accuracy,
        Sensitivity = sensitivity_val,
        Specificity = specificity_val,
        AUC = NA,  # No direct probability estimates for standard ROC curve in simple majority vote
        stringsAsFactors = FALSE
      )
      
      # Update metrics table
      values$metrics <- rbind(values$metrics[values$metrics$Model != "Ensemble (Majority Vote)", ], new_metrics)
      
      # Display result
      output$ensemble_result <- renderPrint({
        cat("Ensemble Model Results (Majority Vote):\n\n")
        cat("Models included in ensemble:", paste(available_models, collapse = ", "), "\n\n")
        cat("Confusion Matrix:\n")
        print(conf_matrix)
        cat("\nAccuracy:", round(accuracy, 4), "\n")
        cat("Sensitivity:", round(sensitivity_val, 4), "\n")
        cat("Specificity:", round(specificity_val, 4), "\n")
      })
    })
  })
}

# Run the application
shinyApp(ui, server)