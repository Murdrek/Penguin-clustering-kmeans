## Project Overview
This repository contains an end-to-end Unsupervised Machine Learning project focused on the automated segmentation of penguin populations. Using physical and environmental traits, the goal is to discover underlying patterns and group the populations without human intervention or pre-existing labels.

## Key Features & Workflow
* **Exploratory Data Analysis (EDA):** Data cleaning, feature scaling, and visualization of morphometric measurements.
* **Algorithm Implementation:** Custom Python functions to automate iteration over multiple cluster scales ($K$).
* **Hyperparameter Optimization:** Strategic selection of the optimal number of clusters using the **Elbow Method (Inertia)** and **Silhouette Coefficient** analysis.
* **Insights Generation:** Profiling the resulting clusters to understand the distinct biological groups discovered by the model.

## Tech Stack
* Python
* Scikit-Learn (KMeans, Silhouette Score)
* Pandas & NumPy
* Seaborn & Matplotlib (Data Visualization)
