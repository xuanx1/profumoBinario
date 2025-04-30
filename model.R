install.packages("gmodels")
install.packages("pROC")

library(tidyverse)  
library(gmodels)
library(psych)
library(car)
library(pROC)


library(randomForest)    # Random Forest model
library(caret)           # Model training and evaluation
library(umap)            # UMAP dimensionality reduction
library(Rtsne)           # t-SNE dimensionality reduction
library(plotly)          # Interactive 3D plots
library(cluster)         # Clustering and silhouette analysis
library(viridis)         # Color palettes
library(corrplot)        # Correlation visualization
library(factoextra)      # PCA visualization


profumo <- read.csv("final_fra2.csv")
view(profumo)

#log model
log.pseudo.r2 <- function(LogModel) {
  dev <- LogModel$deviance
  nullDev <- LogModel$null.deviance
  modelN <- length(LogModel$fitted.values)
  R.l <- 1 - dev/nullDev
  R.cs <- 1 - exp((dev - nullDev)/modelN)
  R.n <- R.cs/(1 - exp(-nullDev/modelN))
  cat("Pseudo R^2 for Logistic Regression\n")
  cat("Cohen R^2               ", round(R.l, 3), "\n")
  cat("Cox and Snell R^2       ", round(R.cs, 3), "\n")
  cat("Nagelkerke R^2          ", round(R.n, 3), "\n")
}



# Split data into training and test sets (80/20)
train_index <- createDataPartition(profumo$Gender_encoded, p = 0.8, list = FALSE)
train_data <- profumo[train_index, ]
test_data <- profumo[-train_index, ]

# Create a log pseudo R-squared function
log.pseudo.r2 <- function(model) {
  # Extract log-likelihoods
  ll_full <- logLik(model)
  ll_null <- logLik(update(model, . ~ 1))
  
  # Calculate McFadden's R-squared
  r2 <- 1 - (ll_full / ll_null)
  return(as.numeric(r2))
}

#remove unisex for binary
train_data <- subset(train_data, Gender_encoded != 2)


# MODELS

perfume_1 <- glm(
  Gender_encoded ~ price_normalized + scent_strength, 
  family = binomial, 
  data = train_data)
summary(perfume_1)

# add notes_diversity
perfume_2 <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity, 
  family = binomial, 
  data = train_data)
summary(perfume_2)

# add 3x  base notes
perfume_3 <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity + 
    base_note_0 + base_note_1 + base_note_2, 
  family = binomial, 
  data = train_data)
summary(perfume_3)

# add 3x middle notes
perfume_4 <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity + 
    middle_note_0 + middle_note_1 + middle_note_2, 
  family = binomial, 
  data = train_data)
summary(perfume_4)

# add perfume concentration
perfume_5 <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity, 
  family = binomial, 
  data = train_data)
summary(perfume_5)


perfume_final <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity + 
    middle_note_0 + middle_note_1 + base_note_0, 
  family = binomial, 
  data = train_data)
summary(perfume_final)

# Alternative final model
perfume_final2 <- glm(
  Gender_encoded ~ scent_strength + notes_diversity + middle_note_0, 
  family = binomial, 
  data = train_data)
summary(perfume_final2)


# (2) Run Diagnostics
exp(coef(perfume_final))
log.pseudo.r2(perfume_final)

# (3) Insert model probability into our dataset
test_data$prob_final <- predict(best_model, newdata = test_data, type = "response")
quantile(test_data$prob_final)

# (4) Predict gender based on model
test_data$pred_final <- ifelse(test_data$prob_final > 0.5, 1, 0)
test_data$pred_final <- factor(test_data$pred_final, levels = levels(test_data$Gender_encoded))

# How many false positives/negatives do you have?
table(test_data$pred_final)

# Calculate your prediction accuracy
accur_final <- 1 - mean(test_data$pred_final != test_data$Gender_encoded)

# Create the confusion matrix
conf_matrix <- table(Predicted = test_data$pred_final, Actual = test_data$Gender_encoded)

# Plot the confusion matrix
library(ggplot2)

conf_df <- as.data.frame(conf_matrix)
colnames(conf_df) <- c("Actual", "Predicted", "Freq")

ggplot(conf_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  labs(title = "Confusion Matrix for Perfume Gender Prediction",
       x = "Predicted Gender",
       y = "Actual Gender",
       fill = "Frequency") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_minimal()
ggsave("perfume_confusion_matrix.png", width = 8, height = 6)


# Add a column for the actual outcome
test_data$outcome <- test_data$Gender_encoded

# Plot the distribution of predicted probabilities
ggplot(test_data, aes(x = prob_final, fill = outcome)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Predicted Probabilities for Gender",
       x = "Predicted Probability of Being Feminine",
       y = "Density",
       fill = "Actual Gender") +
  theme_minimal()
ggsave("perfume_probability_distribution.png", width = 10, height = 6)


# Create a data frame for metrics
metrics_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  Value = c(accur_final, precision, recall, f1_score)
)

# Plot the metrics
ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5) +
  labs(title = "Model Performance Metrics",
       y = "Value",
       x = "") +
  theme_minimal() +
  scale_y_continuous(limits = c(0, 1)) +
  geom_text(aes(label = round(Value, 3)), vjust = -0.5, size = 5)
ggsave("perfume_model_metrics.png", width = 10, height = 6)




# Feature importance based on coefficients
coef_df <- data.frame(
  Feature = names(coef(best_model)),
  Coefficient = coef(best_model),
  Abs_Coefficient = abs(coef(best_model))
)

# Sort by absolute coefficient value (importance)
coef_df <- coef_df %>%
  arrange(desc(Abs_Coefficient))

# Plot feature importance
ggplot(coef_df, aes(x = reorder(Feature, Abs_Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance in Gender Prediction",
       x = "Feature",
       y = "Coefficient Value")
ggsave("perfume_feature_importance.png", width = 10, height = 6)

# Dimensionality Reduction for Visualization
# PCA for visualization
pca_data <- prcomp(scale(processed_data[, -which(names(processed_data) == "Gender_encoded")]), 
                   center = TRUE, scale. = TRUE)
pca_scores <- as.data.frame(pca_data$x[, 1:3])
pca_scores$Gender <- processed_data$Gender_encoded

# Create 3D PCA plot
pca_plot <- plot_ly(pca_scores, x = ~PC1, y = ~PC2, z = ~PC3, 
                    color = ~Gender, 
                    colors = c("#440154", "#21908C"),
                    type = "scatter3d", mode = "markers",
                    marker = list(size = 5, opacity = 0.7)) %>%
  layout(title = "3D PCA Plot of Perfume Data by Gender",
         scene = list(xaxis = list(title = "PC1"),
                      yaxis = list(title = "PC2"),
                      zaxis = list(title = "PC3")))
htmlwidgets::saveWidget(pca_plot, "perfume_pca_3d_plot.html")

# Summary
cat("\n--- PERFUME GENDER PREDICTION SUMMARY ---\n")
cat("Best model:", best_model_name, "\n")
cat("Model formula:", deparse(formula(best_model)), "\n")
cat("Accuracy:", round(accur_final, 5), "\n")
cat("F1 Score:", round(f1_score, 5), "\n")
cat("Pseudo R-squared:", round(log.pseudo.r2(best_model), 5), "\n\n")

cat("Top 5 important features for gender prediction:\n")
for(i in 1:min(5, nrow(coef_df))) {
  cat("  ", i, ". ", coef_df$Feature[i], " (coefficient: ", round(coef_df$Coefficient[i], 3), ")\n")
}
cat("\n")

cat("Output files created:\n")
cat("  - perfume_confusion_matrix.png\n")
cat("  - perfume_probability_distribution.png\n")
cat("  - perfume_model_metrics.png\n")
cat("  - perfume_feature_importance.png\n")
cat("  - perfume_pca_3d_plot.html\n\n")

cat("--- END OF ANALYSIS ---\n")

