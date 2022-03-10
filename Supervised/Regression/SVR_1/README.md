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
**Grid Search**
- techniq to examine some paramater sekaligus
- term of tuning parameter to increase performance ML Model

**Code**
- build model with parameter
  ```py
  from sklearn.model_selection import GridSearchCV
  
  # membangun model dengan parameter C, gamma, dan kernel
  
  model = SVR()
  parameters = {
      'kernel': ['rbf'],
      'C':     [1000, 10000, 100000],
      'gamma': [0.5, 0.05,0.005]
  }
  grid_search = GridSearchCV(model, parameters)
  ```
- show best parameter
  ```py
  print(grid_search.best_params_)
  ```

- addition, rebuild model from gridSearch result 
  ```py
  # membuat model SVM baru dengan parameter terbaik hasil grid search
  model_baru  = SVR(C=100000, gamma=0.005, kernel='rbf')
  model_baru.fit(X,y)
  ```
