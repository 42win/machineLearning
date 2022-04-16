# Classification multiClass Iri

## Description

- Dataset [link](https://www.kaggle.com/uciml/iris)


## Code

- model architect
  ```py
  model = Sequential([    
                    Dense(64, activation='relu', input_shape=(4,)),    
                    Dense(64, activation='relu'),    
                    Dense(3, activation='softmax'),])
  ```

- model compile
  ```py
  model.compile(optimizer='Adam',
                loss='categorical_crossentropy',
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
