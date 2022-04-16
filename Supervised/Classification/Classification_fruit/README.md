# Orange or Jeruk Classification

## Model
- Model JST Sequential
  

## Description

- Dataset [link](https://www.kaggle.com/joshmcadams/oranges-vs-grapefruit)


## Code

- normalization
  ```py
  # Normalization
  from sklearn import preprocessing
  min_max_scaler = preprocessing.MinMaxScaler()
  X_scale = min_max_scaler.fit_transform(X)
  X_scale
  ```

- convert to float32
  ```py
  Y_train.astype(np.float32)
  ```

- model architect
  ```py
  model = Sequential([    
                      Dense(32, activation='relu', input_shape=(5,)),    
                      Dense(32, activation='relu'),    
                      Dense(1, activation='sigmoid'),])
  ```

- model compile
  ```py
  model.compile(optimizer='sgd',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  ```
- model fit
  ```py
  model.fit(X_train, Y_train, epochs=100)
  ```

- model evaluate
  ```py
  model.evaluate(X_test, Y_test)
  ```

## Note
  
  - JST can't process string so we have to convert it to numeric
    ```py
    df.name[df.name == 'orange'] = 0
    ```

  - Jst can't process dataframe so we have to convert it to numpy array
    ```py
    dataset = df.values
    ```
