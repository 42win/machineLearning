# Clasification image of number

## Library : 
  - tensorflow
  - keras
  - numpy
  - matplotlib

## Description

- **Type**  : supervised - classification

- **Background** : if we see number, we can easly know what number it is, but it is difficult for blind person
  
- **Bussiness Understanding**
  1. Problem Statement
     - How to make model to classify fashion items ?
     - How does the accuracy of the prediction model result ? 
  
  2. Goals 
     - to know how to make model to classify number pictures
     - to know accuration result of the model
  
- **Data Understanding**
  
  1. Overview
   
       - we using tensorflow dataset tf.data.datasets. <br> to know more about dataset [link]([tf.data.datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets))
       - it is images dataset which consist of 60.000 data training and 10.000 data testing. each image is 28 x 28 gray-scale picture
  
       - Label on Datasets <br> 0 1 2 3 4 5 6 7 8 9 

    2. Data Preparation
   
       - Train Test Split
            ```py
            (gambar_latih, label_latih), (gambar_testing, label_testing) = mnist.load_data()
            ```
        - Data Normalisation <br>
          in this case, Normalisation have function to minimalize computation cost by changed pixel value to in range [0,1]
            ```py
            train_images = train_images / 255.0
            test_images = test_images / 255.0
            ``` 

    3. Modeling

        - Model:
          - ANN 
          - JST Model : Making MLP (Multi Layer Perceptron) using keras library by calling ``tf.keras.models.Sequential`` and we make 3 layers
            1. input layer
               - using flatten layer (to change array 2 -> 1 dimensi)
            2. hidden layer
                - neuron : 128
                - using activation function : relu
            3. outpur layer
                - using activation function : softmax
                - return 10 classes, each node contains skor
          - model settings :
            - loss function : sparse_categorical_crossentropy
            - optimizer : Adam()
            - metrics : accuracy

            ```py
            model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), 
                                                tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                                tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
            ```
    
## Work Stages

1. ensure TF(Tensorflow) is above 2.0 version
2. import and load dataset from tf.keras.datasets
3. normalize data
4. build model using Sequenstial JST Model
5. compile and start train model 
6. evaluate model
7. make predictions using trained model 

## Evaluation

- in this cas we got accuration = 0.98
  
    ```py
    test_loss, test_acc = model.evaluate(gambar_latih,  label_latih, verbose=2)

    print('\nTest accuracy:', test_acc)
    ```

## Code snippet

- import dataset object
  ```py
  mnist = tf.keras.datasets.mnist
  ```

- load dataset
  ```py
  (gambar_latih, label_latih), (gambar_testing, label_testing) = mnist.load_data()
  ```
- image show
  ```py  
  import matplotlib.pyplot as plt

  plt.imshow(gambar_latih[0]) 
  ```

- image normalisasi <br> 
  to change image 255 px to in range 0-1 
  ```py
  gambar_latih  = gambar_latih / 255.0
  gambar_testing = gambar_testing / 255.0
  ```

- multiple image show 
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
        
        #show image and binary format
        plt.imshow(gambar_latih[i], cmap=plt.cm.binary)

        plt.xlabel(label_latih[i])
  plt.show()
  ```

- flattern layer
  ```py
  tf.keras.layers.Flatten(input_shape=(28,28)
  ```
  to change input array 2 dimensions (28x28px) to array 1 dimensions (784px->28*28)

- model prediction
  ```py
  # predicting images
    from tensorflow.keras.preprocessing import image
    img=gambar_testing[6]

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
        
    classes = model.predict(images, batch_size=10)  

    np.argmax(classes) 
  ```
  model predict return array 10 angka sesuai dengan jumlah class so we use ``argmax`` to select maximun predict among class predict
