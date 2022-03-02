# SVR (Support Vector Regression)


**Goal**
- implement SVR to predict salary based on lama kerja

**Tahapan**

1. Ubah data menjadi Dataframe.
2. Pisahkan atribut dan label.
3. Latih model SVR.
4. Buat plot dari model.

**dataset**
- Dataset yang akan kita gunakan adalah data tentang lama kerja seseorang dan gajinya. 
  
- [link](https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression)

**Code**
- svr implementation
  ```py
  from sklearn.svm import SVR
  
  # membangun model dengan parameter C, gamma, dan kernel
  
  #svr use dua parameter C: regulazationParameter, dan E: MarginError
   
  model  = SVR(C=1000, gamma=0.05, kernel='rbf')
  
  # melatih model dengan fungsi fit
  model.fit(X,y)
  ```
