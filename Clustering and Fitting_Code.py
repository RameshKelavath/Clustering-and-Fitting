#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scipy.stats import norm
import warnings


# In[2]:


# Suppress all warnings
warnings.filterwarnings("ignore")
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


# In[3]:


# Display the first few rows of the dataset
print(iris_df.head())


# In[4]:


pip install --upgrade seaborn


# In[5]:


# Function to plot pairwise scatter plots
def plot_pairwise_scatter(data):
    """
    Plot pairwise scatter plots to visualize relationships between features.

    Args:
    data (DataFrame): Input dataset.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))
    sns.pairplot(data, hue='target', palette='viridis', diag_kind='kde')
    plt.suptitle('Pairwise Scatter Plots of Features', y=1.02)
    plt.show()

# Plot pairwise scatter plots
plot_pairwise_scatter(iris_df)


# In[6]:


# Function to plot histograms for each feature
def plot_histograms(data):
    """
    Plot histograms for each feature in the dataset.

    Args:
    data (DataFrame): Input dataset.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(data.columns[:-1]):
        plt.subplot(2, 2, i + 1)
        sns.histplot(data=data, x=feature, kde=True)
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Plot histograms for each feature
plot_histograms(iris_df)


# In[8]:


# Function to perform K-means clustering
def kmeans_clustering(data, n_clusters):
    """
    Perform K-means clustering on the data.

    Args:
    data (DataFrame): Input data for clustering.
    n_clusters (int): Number of clusters.

    Returns:
    KMeans: Fitted K-means model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans


# In[9]:


# Perform K-means clustering
kmeans_model = kmeans_clustering(iris_df.iloc[:, :-1], n_clusters=3)

# Add cluster labels to the dataset
iris_df['cluster'] = kmeans_model.labels_


# In[20]:


# Perform K-means clustering
kmeans_model = kmeans_clustering(iris_df.iloc[:, :-1], n_clusters=3)

# Add cluster labels to the dataset
iris_df['cluster'] = kmeans_model.labels_

# Function to plot K-means clusters
def plot_kmeans_clusters(data, kmeans_model):
    """
    Plot K-means clustering results.

    Args:
    data (DataFrame): Input data.
    kmeans_model: Fitted K-means clustering model.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x=data.columns[0], y=data.columns[1], hue='cluster', palette='viridis', legend='full')
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.legend()
    plt.show()

# Plot K-means clustering results
plot_kmeans_clusters(iris_df, kmeans_model)


# In[12]:


# Function to perform hierarchical clustering
def hierarchical_clustering(data, n_clusters):
    """
    Perform hierarchical clustering on the data.

    Args:
    data (DataFrame): Input data for clustering.
    n_clusters (int): Number of clusters.

    Returns:
    AgglomerativeClustering: Fitted hierarchical clustering model.
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical.fit(data)
    return hierarchical

# Perform hierarchical clustering
hierarchical_model = hierarchical_clustering(iris_df.iloc[:, :-2], n_clusters=3)

# Function to plot hierarchical clustering dendrogram
def plot_dendrogram(data, model):
    """
    Plot the dendrogram for hierarchical clustering.

    Args:
    data (DataFrame): Input data.
    model: Fitted hierarchical clustering model.

    Returns:
    None
    """
    plt.figure(figsize=(12, 8))
    sns.set(style='whitegrid')
    sns.clustermap(data, row_cluster=False, col_cluster=False, cmap='viridis', method='ward')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

# Plot hierarchical clustering dendrogram
plot_dendrogram(iris_df.iloc[:, :-2], hierarchical_model)


# In[13]:


# Function to perform linear regression
def linear_regression(data):
    """
    Perform linear regression on the data.

    Args:
    data (DataFrame): Input data.

    Returns:
    LinearRegression: Fitted linear regression model.
    """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -2]
    model = LinearRegression()
    model.fit(X, y)
    return model

# Fit linear regression model
linear_model = linear_regression(iris_df)

# Function to plot linear regression line
def plot_linear_regression(data, model):
    """
    Plot linear regression line.

    Args:
    data (DataFrame): Input data.
    model: Fitted linear regression model.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=data.columns[0], y=data.columns[-2], color='blue', label='Actual Data')
    plt.plot(data.iloc[:, :-1], model.predict(data.iloc[:, :-1]), color='red', label='Linear Regression')
    plt.title('Linear Regression')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[-2])
    plt.legend()
    plt.show()

# Plot linear regression line
plot_linear_regression(iris_df, linear_model)


# In[18]:


# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Selecting only two features for simplicity
iris_df = iris_df[['petal length (cm)', 'sepal width (cm)', 'target']]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_df[['petal length (cm)']], iris_df['sepal width (cm)'], test_size=0.2, random_state=42)
# Function to perform polynomial regression
def polynomial_regression(X_train, y_train, degree):
    """
    Perform polynomial regression on the data.

    Args:
    X_train (DataFrame): Input features for training.
    y_train (Series): Target variable for training.
    degree (int): Degree of the polynomial.

    Returns:
    LinearRegression: Fitted polynomial regression model.
    """
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    return model, polynomial_features

# Fit polynomial regression model
polynomial_model, polynomial_features = polynomial_regression(X_train, y_train, degree=3)

# Function to plot polynomial regression curve
def plot_polynomial_regression(X_train, y_train, model, features):
    """
    Plot polynomial regression curve.

    Args:
    X_train (DataFrame): Input features for training.
    y_train (Series): Target variable for training.
    model: Fitted polynomial regression model.
    features: Polynomial features.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=iris_df, x='petal length (cm)', y='sepal width (cm)', color='blue', label='Actual Data')
    x_values = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    x_poly = features.transform(x_values)
    plt.plot(x_values, model.predict(x_poly), color='red', label='Polynomial Regression')
    plt.title('Polynomial Regression')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    plt.show()

# Plot polynomial regression curve
plot_polynomial_regression(X_train, y_train, polynomial_model, polynomial_features)


# In[15]:


# Function to calculate silhouette score for K-means clustering
def calculate_silhouette_score(data, max_clusters):
    """
    Calculate silhouette score to determine the optimal number of clusters for K-means.

    Args:
    data (DataFrame): Input data.
    max_clusters (int): Maximum number of clusters to try.

    Returns:
    None
    """
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, cluster_labels))

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for K-means Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Calculate silhouette score for K-means clustering
calculate_silhouette_score(iris_df.iloc[:, :-2], max_clusters=10)


# In[16]:


# Function to plot elbow method for K-means clustering
def plot_elbow_method(data, max_clusters):
    """
    Plot the elbow method to determine the optimal number of clusters for K-means.

    Args:
    data (DataFrame): Input data.
    max_clusters (int): Maximum number of clusters to try.

    Returns:
    None
    """
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method for K-means Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()

# Plot elbow method for K-means clustering
plot_elbow_method(iris_df.iloc[:, :-2], max_clusters=10)


# In[ ]:




