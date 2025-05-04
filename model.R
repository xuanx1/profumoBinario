install.packages("gmodels")
install.packages("pROC")
install.packages("fastDummies")

library(tidyverse)  
library(gmodels)
library(psych)
library(car)
library(pROC)
library(fastDummies)

library(randomForest)   
library(caret)           # Model training and evaluation
library(umap)            # UMAP dimensionality reduction
library(Rtsne)           # t-SNE dimensionality reduction
library(plotly)          # Interactive 3D plots
library(cluster)         # Clustering and silhouette analysis
library(viridis)         # Color palettes
library(corrplot)        # Correlation visualization
library(factoextra)      # PCA visualization


profumo <- read.csv("final_fra3.csv")
view(profumo)

#crosstables
CrossTable(profumo$Gender_encoded, profumo$scent_strength, expected = TRUE, format = "SPSS")

CrossTable(profumo$Gender_encoded, profumo$price_normalized, expected = TRUE, format = "SPSS")

CrossTable(profumo$Gender_encoded, profumo$notes_diversity, expected = TRUE, format = "SPSS")

CrossTable(profumo$price_normalized, profumo$notes_diversity, expected = TRUE, format = "SPSS")


#histogram #ff6056", "#4181f2, no unisex and kids
profumo <- subset(profumo, Gender_encoded != 2)

ggplot(profumo, aes(x = gender)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Gender", x = "Gender", y = "Count") +
  theme_minimal()

ggplot(profumo, aes(x = concentration)) +
  geom_bar(fill = "skyblue", color = "orange") +
  labs(title = "Distribution of concentration", x = "concentration", y = "Count") +
  theme_minimal()

ggplot(profumo, aes(x = price)) +
  geom_bar(fill = "skyblue", color = "red") +
  labs(title = "Distribution of price", x = "price", y = "Count") +
  theme_minimal()

ggplot(profumo, aes(x = notes_diversity)) +
  geom_bar(fill = "skyblue", color = "pink") +
  labs(title = "Distribution of notes diversity", x = "notes count", y = "Count") +
  theme_minimal()



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


#remove unisex for binary
train_data <- subset(train_data, Gender_encoded != 2)


# MODELS

perfume_1 <- glm(
  Gender_encoded ~ price_normalized + scent_strength, 
  family = binomial, 
  data = train_data)
summary(perfume_1)

exp(coef(perfume_1))
log.pseudo.r2(perfume_1)


# add notes_diversity
perfume_2 <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity, 
  family = binomial, 
  data = train_data)
summary(perfume_2)

exp(coef(perfume_2))
log.pseudo.r2(perfume_2)


# add dummy coded scents
perfume_3 <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity + Woody + Floral, 
  family = binomial,
  data = train_data)
summary(perfume_3)

exp(coef(perfume_3))
log.pseudo.r2(perfume_3)


# add exclusive notes - columns with "clary sage", "fir", "peony", "jasmine sambac" notes as dummy model
train_data$clary_sage_present <- apply(train_data[, grep("base_note_|middle_note_", names(train_data))], 1, function(row) {
  any(grepl("clary sage", row, ignore.case = TRUE))
})
train_data$clary_sage_present <- as.numeric(train_data$clary_sage_present)

note_columns <- grep("base_note_|middle_note_", names(train_data), value = TRUE)

train_data$fir_present <- apply(train_data[, note_columns], 1, function(row) {
  any(grepl("fir", row, ignore.case = TRUE))
})

train_data$peony_present <- apply(train_data[, note_columns], 1, function(row) {
  any(grepl("peony", row, ignore.case = TRUE))
})

train_data$jasmine_sambac_present <- apply(train_data[, note_columns], 1, function(row) {
  any(grepl("jasmine sambac", row, ignore.case = TRUE))
})

train_data$fir_present <- as.numeric(train_data$fir_present)
train_data$peony_present <- as.numeric(train_data$peony_present)
train_data$jasmine_sambac_present <- as.numeric(train_data$jasmine_sambac_present)

perfume_4 <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity + clary_sage_present +
    fir_present + peony_present + jasmine_sambac_present,
  family = binomial,
  data = train_data
)

summary(perfume_4)


exp(coef(perfume_4))
log.pseudo.r2(perfume_4)


# add common notes
note_columns <- grep("base_note_|middle_note_", names(train_data), value = TRUE)

train_data$patchouli_present <- apply(train_data[, note_columns], 1, function(row) {
  any(grepl("patchouli", row, ignore.case = TRUE))
})

train_data$amber_present <- apply(train_data[, note_columns], 1, function(row) {
  any(grepl("amber", row, ignore.case = TRUE))
})

train_data$salwood_present <- apply(train_data[, note_columns], 1, function(row) {
  any(grepl("salwood", row, ignore.case = TRUE))
})

train_data$sandalwood_present <- apply(train_data[, note_columns], 1, function(row) {
  any(grepl("sandalwood", row, ignore.case = TRUE))
})

train_data$musk_present <- apply(train_data[, note_columns], 1, function(row) {
  any(grepl("musk", row, ignore.case = TRUE))
})

note_vars <- c("patchouli_present", "amber_present", "salwood_present", "sandalwood_present", "musk_present")
train_data[note_vars] <- lapply(train_data[note_vars], as.numeric)

perfume_5 <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity +
    patchouli_present + amber_present + salwood_present +
    sandalwood_present + musk_present,
  family = binomial,
  data = train_data
)

summary(perfume_5)

exp(coef(perfume_5))
log.pseudo.r2(perfume_5)


# combine significant notes and scent
perfume_6 <- glm(
  Gender_encoded ~ scent_strength + patchouli_present + musk_present + Woody + Floral, 
  family = binomial, 
  data = train_data)
summary(perfume_6)

exp(coef(perfume_6))
log.pseudo.r2(perfume_6)


# final model
perfume_final <- glm(
  Gender_encoded ~ price_normalized + scent_strength + notes_diversity + patchouli_present + musk_present + Woody + Floral,  
  family = binomial, 
  data = train_data)
summary(perfume_final)


# (2) Run Diagnostics
exp(coef(perfume_final))
log.pseudo.r2(perfume_final)


# (3) Insert model probability into train data
train_data <- train_data[!is.na(train_data$prob_final) & !is.nan(train_data$prob_final), ]

train_data$prob_final <- predict(perfume_final, newdata = train_data, type = "response")
quantile(train_data$prob_final)

# make sure levels in gender
train_data$Gender_encoded <- factor(train_data$Gender_encoded)

# (4) Predict gender based on model
train_data$pred_final <- ifelse(train_data$prob_final > 0.5, 1, 0)
train_data$pred_final <- factor(train_data$pred_final, levels = levels(train_data$Gender_encoded))

names(train_data)
View(train_data)
# export test data and results
write.csv(train_data, "train_profumo.csv", row.names = FALSE)

# How many false positives/negatives do you have?
table(train_data$pred_final)

# Calculate your prediction accuracy
accur_final <- 1 - mean(train_data$pred_final != train_data$Gender_encoded)
accur_final









# dummy code and format test data
test_data <- subset(test_data, Gender_encoded != 2)

test_data$clary_sage_present <- apply(test_data[, grep("base_note_|middle_note_", names(test_data))], 1, function(row) {
  any(grepl("clary sage", row, ignore.case = TRUE))
})
test_data$clary_sage_present <- as.numeric(test_data$clary_sage_present)

note_columns <- grep("base_note_|middle_note_", names(test_data), value = TRUE)

test_data$fir_present <- apply(test_data[, note_columns], 1, function(row) {
  any(grepl("fir", row, ignore.case = TRUE))
})

test_data$peony_present <- apply(test_data[, note_columns], 1, function(row) {
  any(grepl("peony", row, ignore.case = TRUE))
})

test_data$jasmine_sambac_present <- apply(test_data[, note_columns], 1, function(row) {
  any(grepl("jasmine sambac", row, ignore.case = TRUE))
})

test_data$fir_present <- as.numeric(test_data$fir_present)
test_data$peony_present <- as.numeric(test_data$peony_present)
test_data$jasmine_sambac_present <- as.numeric(test_data$jasmine_sambac_present)


note_columns <- grep("base_note_|middle_note_", names(test_data), value = TRUE)

test_data$patchouli_present <- apply(test_data[, note_columns], 1, function(row) {
  any(grepl("patchouli", row, ignore.case = TRUE))
})

test_data$amber_present <- apply(test_data[, note_columns], 1, function(row) {
  any(grepl("amber", row, ignore.case = TRUE))
})

test_data$salwood_present <- apply(test_data[, note_columns], 1, function(row) {
  any(grepl("salwood", row, ignore.case = TRUE))
})

test_data$sandalwood_present <- apply(test_data[, note_columns], 1, function(row) {
  any(grepl("sandalwood", row, ignore.case = TRUE))
})

test_data$musk_present <- apply(test_data[, note_columns], 1, function(row) {
  any(grepl("musk", row, ignore.case = TRUE))
})

note_vars <- c("patchouli_present", "amber_present", "salwood_present", "sandalwood_present", "musk_present")
test_data[note_vars] <- lapply(test_data[note_vars], as.numeric)





# (3) Insert model probability into our dataset
test_data <- test_data[!is.na(test_data$prob_final) & !is.nan(test_data$prob_final), ]
test_data$prob_final <- predict(perfume_final, newdata = test_data, type = "response")
quantile(test_data$prob_final)

# make sure levels in gender
test_data$Gender_encoded <- factor(test_data$Gender_encoded)

# (4) Predict gender based on model
test_data$pred_final <- ifelse(test_data$prob_final > 0.5, 1, 0)
test_data$pred_final <- factor(test_data$pred_final, levels = levels(test_data$Gender_encoded))

names(test_data)
View(test_data)
# export test data and results
write.csv(test_data, "test_profumo.csv", row.names = FALSE)

# How many false positives/negatives do you have?
table(test_data$pred_final)

# Calculate your prediction accuracy
accur_final <- 1 - mean(test_data$pred_final != test_data$Gender_encoded)

accur_final



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


# Extract values from the confusion matrix
TP <- conf_matrix["1", "1"]  # True Positives
FP <- conf_matrix["1", "0"]  # False Positives
FN <- conf_matrix["0", "1"]  # False Negatives
TN <- conf_matrix["0", "0"]  # True Negatives

# Calculate Precision and Recall
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

# Calculate F1 Score
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the results
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")


#
install.packages("pROC")
library(pROC)

perfume_test <- read.csv("test_profumo2.csv")

# Create an ROC curve with roc(RESPONSE, PREDICTOR):
roc_final <- roc(perfume_test$Gender_encoded, perfume_test$prob_final)

# Plot and legend:
plot(roc_final, main = "ROC Curve, Gender Prediction") # main adds a title

abline(a = 0, b = 1, lty = 2)

legend(
  "bottomright", 
  legend = paste0("AUC = ", round(auc(roc_final), 2)),
  bty = "n", 
  cex = 0.8
)
$prob_final)

# Plot and legend:
plot(roc_final, main = "ROC Curve, Gender Prediction") # main adds a title

abline(a = 0, b = 1, lty = 2)

legend(
  "bottomright", 
  legend = paste0("AUC = ", round(auc(roc_final), 2)),
  bty = "n", 
  cex = 0.8
)


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
  Feature = names(coef(perfume_final)),
  Coefficient = coef(perfume_final),
  Abs_Coefficient = abs(coef(perfume_final))
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
numeric_data <- profumo[, sapply(profumo, is.numeric)]
numeric_data <- na.omit(numeric_data)
numeric_data <- numeric_data %>%
  mutate_all(~ ifelse(is.infinite(.), NA, .))  # Replace Inf/-Inf with NA
numeric_data <- numeric_data %>%
  mutate_all(~ ifelse(is.na(.), mean(., na.rm = TRUE), .))  # Impute NA

# PCA for visualization
pca_data <- prcomp(scale(numeric_data[, -which(names(numeric_data) == "Gender_encoded")]), 
                   center = TRUE, scale. = TRUE)
pca_scores <- as.data.frame(pca_data$x[, 1:3])
pca_scores$Gender <- numeric_data$Gender_encoded

# Create 3D PCA plot
pca_plot <- plot_ly(pca_scores, x = ~PC1, y = ~PC2, z = ~PC3, 
                    color = ~Gender, 
                    colors = c("#ff6056", "#4181f2"),
                    type = "scatter3d", mode = "markers",
                    marker = list(size = 5, opacity = 0.7)) %>%
  layout(title = "3D PCA Plot of Perfume Data by Gender",
         scene = list(xaxis = list(title = "PC1"),
                      yaxis = list(title = "PC2"),
                      zaxis = list(title = "PC3")))
htmlwidgets::saveWidget(pca_plot, "perfume_pca_3d_plot.html")


best_model_name <- "Model 7"


# Summary
cat("\n--- PERFUME GENDER PREDICTION SUMMARY ---\n")
cat("Best model:", best_model_name, "\n")
cat("Model formula:", deparse(formula(perfume_final)), "\n")
cat("Accuracy:", round(accur_final, 5), "\n")
cat("F1 Score:", round(f1_score, 5), "\n")
cat("Pseudo R-squared:", round(log.pseudo.r2(perfume_final), 5), "\n\n")

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
