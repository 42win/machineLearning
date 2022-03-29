# Clasification image of fashion


## Libary

  - Tensorflow
  - numpy
  - matplotlib
  - keras

## Description

- **Type**  : supervised - classification

- **Background** : in fashion industry, there are a lot of items in and employess need time to seperate the items by type (t-shirt, trouser, etc)
  
- **Bussiness understanding** : 
  1. Problem Statement
      - How to make model to classify fashion items ?
      - How does the accuracy of the prediction model result ? 

  2. Goals 
      - to know how to make model to classify fashion items
      - to know accuration result of the model

- **Data Understanding**

  1. Overview  
     - we using tensorflow dataset tf.data.datasets, Fashion-MNIST  . <br> to know more about dataset [link]([tf.data.datasets](https://github.com/zalandoresearch/fashion-mnist))
     - it is images dataset which consist of 60.000 data training and 10.000 data testing. each image is 28 x 28 gray-scale picture

     - Label on datasets
  
        | Label | Keterangan |
        | :--: | :-- |
        | 0 | T-Shirt
        | 1 | Trouser
        | 2 | Pullover
        | 2 | Dress
        | 3 | Coat
        | 4 | Sandal
        | 5 | Shirt
        | 6 | Sneaker
        | 7 | Bag
        | 8 | Ankle Boot


  2. Data Preparation
   
     - Train Test split : 
        ```py
        (gambar_latih, label_latih), (gambar_testing, label_testing) = mnist.load_data()
        ```
        we split data when we load it from source
  
     - Data Normalisation : <br>
       in this case, Normalisation have function to minimalize computation cost by changed pixel value to in range [0,1]
        ```py
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        ``` 
     
     - Modeling
       - ANN 
       - JST Model : Making MLP (Multi Layer Perceptron) using keras library by calling ``tf.keras.models.Sequential`` and we make 3 layers
         1. input layer
            - we use 28x28 pixel as input shape based on resolution of image sources
            - using flatten layer (to change array 2 -> 1 dimensi)
         2. hidden layer
             - neuron : 128
             - using activation function : relu
             - ReLu (default activation func. for ANN)
         3. outpur layer
             - using activation function : softmax (for more than 3 classes )
             - return 10 classes, each node contains skor
  
            ```py
            model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), 
                                                tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                                tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
            ```

       - model settings :  
         
         before model ready to train. it need several setting again. this add in compile stage
            ```py
            model.compile(optimizer = tf.optimizers.Adam(),
                            loss = 'sparse_categorical_crossentropy',
                            metrics=['accuracy'])
            ```
         - loss function : sparse_categorical_crossentropy (for more than 3 classes )
         - optimizer : Adam()
         - metrics : accuracy
  
## Work Stages

1. import library
2. import and load dataset from tf.keras.datasets
3. explore data
4. preprocess data : normalisation
5. build model 
6. compile and train model
7. evaluate model
8. make predictions using trained model 
9. verify model

## Evaluation

- in this case we got accuration = 0.88

    ```py
    test_loss, test_acc = model.evaluate(gambar_latih,  label_latih, verbose=2)

    print('\nTest accuracy:', test_acc)
    ```
  
## Code Snippet

- Make array class names variabel based on test labels position
  ```py
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  ```

- check array length ``len(variabelName)``

- check image pixel 
  ```py

  import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
  ```

- show multiple images with its label
  ```py
  plt.figure(figsize=(10,10))

    for i in range(25):

        #to show multiple plot subplot(row,colum)
        plt.subplot(5,5,i+1)

        #to hide number location in line x and y
        plt.xticks([])
        plt.yticks([])

        #to hide grid line in box
        plt.grid(False)

        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    
    plt.show()
  ```

- to get index of maximun value
  ```py
  np.argmax(predictions[0])
  ``` 

- make prediction
  ```py
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    np.argmax(predictions[0])
  ```
