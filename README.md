# SupPCA
Supervised Principal Component Analysis Codes

Supervised PCA (SupPCA) framework that extends PCA to incorporate auxiliary data. The auxiliary information that potentially drives the underlying structure of the primary data of interest is referred to as supervision. The goal is to obtain a more interpretable and accurate low-rank approximation of the primary data with the help of supervision.
 
To better accommodate high dimensional data and functional data, we also extend the supervised PCA framework to incorporate regularization, and develop a supervised regularized PCA method (SupSFPCA). Smoothness and sparsity constraints are imposed on loading vectors to reduce variability and enhance interpretability. In addition, we also impose sparsity on supervision coeffcients to identify auxiliary variables with no supervision effect.