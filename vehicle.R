# Load libraries
library(readr)
library(dplyr)
library(ggplot2)
library(FactoMineR)
library(factoextra)
library(caret)
library(cluster)

# Load dataset
df <- read_csv("C:/Users/Admin/OneDrive/Desktop/vehicles.csv")

# ---- Data Exploration ----
cat("Shape of dataset:", dim(df), "\n")
head(df)
cat("\nMissing values per column:\n")
print(colSums(is.na(df)))
cat("\nData Summary:\n")
print(summary(df))

# ---- Data Preprocessing ----
if ("class" %in% colnames(df)) {
  X <- df %>% select(-class)
} else {
  X <- df[, -ncol(df)]
}
X <- na.omit(X)
X_scaled <- scale(X)

# ---- PCA ----
pca_result <- prcomp(X_scaled)
cat("\nExplained Variance Ratio (PC1 & PC2):\n")
print(summary(pca_result)$importance[2, 1:2])

# Plot PCA
fviz_pca_ind(pca_result,
             geom.ind = "point",
             pointshape = 21,
             col.ind = "steelblue",
             repel = TRUE) +
  ggtitle("PCA - Vehicle Dataset")

# ---- Elbow Method ----
fviz_nbclust(X_scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2) +
  labs(title = "Elbow Method for K", x = "Number of clusters", y = "Total Within Sum of Squares")

# ---- K-Means Clustering ----
set.seed(42)
k <- 4
km <- kmeans(X_scaled, centers = k)

# PCA plot with clusters
pca_df <- as.data.frame(pca_result$x[, 1:2])
pca_df$Cluster <- as.factor(km$cluster)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "K-Means Clustering on PCA-Reduced Data") +
  theme_minimal()

# Cluster Sizes
cat("\nCluster Sizes (K-Means):\n")
print(table(km$cluster))

# PCA Loadings
cat("\nFeature Influence on Principal Components (Loadings):\n")
print(pca_result$rotation[, 1:2])

# Compare Clusters to True Labels (if available)
if ("class" %in% colnames(df)) {
  pca_df$TrueLabel <- df$class[rownames(pca_df)]
  cat("\nCluster vs True Label (K-Means):\n")
  print(table(pca_df$Cluster, pca_df$TrueLabel))
}

# ---- Hierarchical Clustering ----
dist_matrix <- dist(X_scaled)
hc <- hclust(dist_matrix, method = "ward.D2")

# Plot Dendrogram
plot(hc, labels = FALSE, main = "Hierarchical Clustering Dendrogram")

# Cut tree into clusters
num_clusters <- 4
hc_clusters <- cutree(hc, k = num_clusters)

df_clean <- df[complete.cases(df[, colnames(X)]), ]
df_clean$Cluster <- as.factor(hc_clusters)

# Cluster Sizes (Hierarchical)
cat("\nCluster Sizes (Hierarchical):\n")
print(table(df_clean$Cluster))

# Compare with 'class' if available
if ("class" %in% colnames(df)) {
  cat("\nCluster vs True Label (Hierarchical):\n")
  print(table(df_clean$Cluster, df_clean$class))
}
