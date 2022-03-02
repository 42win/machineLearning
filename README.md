# Base Command

**Pandas** 

- ``import pandas as pd``

- read file : ``df = pd.read_csv('pathFile/NameFile.csv')``
- see first 5 row : ``df.head()``
- cek null, column, type data, total rows : ``df.info()``
- delete column : ``data = df.drop(columns['columnName'])``
- rename column : ``df.rename(columns={'columnName': newColumnName, '':'' , ..})``
- replace value : ``df['columnName'].replace(['female','male'], [0,1], inplace=True)``
- select colum : 
  - ``df.['ColumnName']``
  - ``df.columns[8]`` only column 8
  - ``df.columns[:8]`` column 0 to 8
  - ``df[4:6]`` row 4 to 6
  

**Data Preprocessing**
- Standarisasi (menyamakan skala each attribute)
  ```py
  from sklearn.preprocessing import StandardScaler
  
  # standarisasi nilai-nilai dari dataset
  scaler = StandardScaler()
  scaler.fit(X)
  X = scaler.transform(X)
  ```

  - inverse skala
  ```py
  X_inverse = scaler.inverse_transform(X)
  X_inverse
  ```

- Divide data test dan training
  ```py
  from sklearn.model_selection import train_test_split
  
  # memisahkan data untuk training dan testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  ```

**Save and Load Model**

- Method 1 - Pickle
    ```py
    import pickle
    
    #save
    with open('iris_pred_model','wb') as f:
        pickle.dump(tree_model,f)

    #load
    with open('iris_pred_model','rb') as f:
    model = pickle.load(f)

    #Test
    print(model.predict([[6.2, 3.4, 5.4, 2.3]])[0])

    ``` 
- Method 2 - Joblib
    ```py
    
    #Save
    import joblib
    joblib.dump(tree_model,"iris_pred_model_1.pkl")

    #Load
    model_1 = joblib.load('iris_pred_model_1.pkl')

    #Test
    print(model_1.predict([[6.2, 3.4, 5.4, 2.3]])[0])
    ```

**Others**

- implementasi to dataset : ``.fit()``

- cek accuracy
  - cara 1
    ```py
    from sklearn.metrics import accuracy_score
    y_pred = model_pertama.predict(X_test)
    
    acc_secore = round(accuracy_score(y_pred, y_test), 4) # 4 jumlah attribut
    
    print('Accuracy: ', acc_secore)
    ```
  - cara 2
    ```py
    model_pertama.score(X_test, y_test)
    ```
