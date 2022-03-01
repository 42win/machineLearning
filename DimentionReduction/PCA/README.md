# PCA (Principal Componen Analysis)

**tahapan**

1. Bagi dataset.
2. Latih model tanpa PCA.
3. Latih model dengan PCA.
4. Evaluasi hasil kedua model.

**Dataset** : dataset iris

**Code**
- create PC object 
  ```py
  pca = PCA(n_components=4) # 4 jumlah attribut
  ```
- implement PCA to dataset
  ```py
  pca_attributes = pca.fit_transform(X_train)
  ```
- show variances each attribute
  ```py
  pca.explained_variance_ratio_
  # output
  # array([0.92848323, 0.04764372, 0.01931005, 0.004563  ]) 
  ```

  hasil dari variance dapat menjadi acuan penentuan pengurangan attribut

