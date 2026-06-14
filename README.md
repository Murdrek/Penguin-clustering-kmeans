# Automated Penguin Population Segmentation / Segmentación Automatizada de Poblaciones de Pingüinos

## 🇺🇸 Project Overview
This repository contains an end-to-end **Unsupervised Machine Learning** project focused on the automated segmentation of penguin populations. Using morphometric measurements, the goal is to discover underlying biological patterns and group the populations without pre-existing labels.

### 🎯 Project Objective
To develop an unsupervised learning model capable of autonomously grouping and segmenting biological populations using continuous physical traits, validating the presence of latent patterns such as species differentiation and sexual dimorphism.

### 📈 Methodology and Process (PACE Framework)
* **Plan & Preprocess:** Irrelevant variables (such as geographic location) were cleaned and isolated. A rigorous data standardization (scaling) was applied to ensure that large-magnitude variables (like body mass in grams) would not bias the mathematical calculations over smaller-scale features.
* **Model & Technical Evaluation:** The data structure was evaluated using cross-optimization techniques. Although global cohesion heuristic indicators (such as the *Silhouette Coefficient*) suggested an optimal split into 4 macro-groups, a **6-cluster model** was selected to maximize operational granularity.

### 🔍 Key Results & Findings
* **Clustering Precision:** The K-means model demonstrated an outstanding ability to identify the natural structure of the data with a negligible margin of error.
* **Pattern Discovery:** The 6-cluster model did not distribute the data chaotically; instead, it perfectly isolated the 3 species (*Adelie, Gentoo, Chinstrap*) and subdivided each one exactly into males and females based exclusively on their physical dimensions.

### 💼 Stakeholder Conclusion
This project successfully demonstrates that unsupervised cluster analysis is a powerful and reliable tool for discovering complex classifications and detailed population profiles without the need for prior human intervention or labeling. This methodology is perfectly scalable to customer segmentation or market product categorization problems.

### 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (`KMeans`, `silhouette_score`, `StandardScaler`)
* **Data & Viz:** Pandas, NumPy, Matplotlib, Seaborn

### 📂 Repository Structure

├── Code/
│   └── Penguin-segmentation-kmeans.py      # Executable Python script with the modeling pipeline
├── Data/
│   └── penguins.csv                        # Source biological dataset
├── Notebooks/
│   └── Penguin-segmentation-kmeans.ipynb   # Interactive data exploration and analysis notebook
└── README.md                               # Project documentation
---

## 🇪🇸 Descripción del Proyecto
Este repositorio contiene un proyecto de **Aprendizaje No Supervisado** enfocado en la segmentación automatizada de poblaciones de pingüinos. Utilizando medidas morfométricas, el objetivo es descubrir patrones biológicos subyacentes y agrupar las poblaciones sin la necesidad de etiquetas preexistentes.

### 🎯 Objetivo del Proyecto
Desarrollar un modelo de aprendizaje no supervisado capaz de agrupar y segmentar poblaciones biológicas de manera autónoma utilizando características físicas continuas, validando la presencia de patrones latentes como la diferenciación de especies y el dimorfismo sexual.

### 📈 Metodología y Proceso (Marco PACE)
* **Planificación y Preprocesamiento:** Se limpiaron y aislaron las variables irrelevantes (como la ubicación geográfica). Aplicamos una estandarización rigurosa de los datos para garantizar que las variables de gran magnitud (como la masa corporal en gramos) no sesgaran los cálculos matemáticos sobre las variables de menor escala.
* **Modelado y Evaluación Técnica:** Se evaluó la estructura de los datos mediante técnicas cruzadas de optimización. Aunque los indicadores heurísticos de cohesión global (como el *Coeficiente de Silueta*) sugerían una división óptima en 4 macro-grupos, se optó por un **modelo de 6 clústeres** para maximizar la granularidad operativa.

### 🔍 Resultados Clave y Hallazgos
* **Precisión del Agrupamiento:** El modelo K-means demostró una capacidad sobresaliente para identificar la estructura natural de los datos con un margen de error insignificante.
* **Descubrimiento de Patrones:** El modelo de 6 clústeres no distribuyó los datos de forma caótica; en su lugar, logró aislar perfectamente a las 3 especies (*Adelie, Gentoo, Chinstrap*) y subdividir cada una de ellas de forma exacta en machos y hembras basándose exclusivamente en sus dimensiones físicas.

### 💼 Conclusión para Stakeholders
El proyecto demuestra con éxito que el análisis de clústeres no supervisado es una herramienta potente y fiable para descubrir clasificaciones complejas y perfiles de población detallados sin necesidad de intervención o etiquetado humano previo. Esta metodología es perfectamente escalable a problemas de de clientes o categorización de productos en el mercado.

### 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python
* **Machine Learning:** Scikit-Learn (`KMeans`, `silhouette_score`, `StandardScaler`)
* **Datos y Visualización:** Pandas, NumPy, Matplotlib, Seaborn

### 📂 Estructura del Repositorio

├── Code/
│   └── Penguin-segmentation-kmeans.py      # Script ejecutable de Python con el pipeline de modelado
├── Data/
│   └── penguins.csv                        # Dataset biológico original
├── Notebooks/
│   └── Penguin-segmentation-kmeans.ipynb   # Notebook interactivo con la exploración y el análisis
└── README.md                               # Documentación del proyecto
