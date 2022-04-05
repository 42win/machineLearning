# Classificaiton image of chess

## Challenge

- image dataset doesnt have same size and ML Model need same size to process it. 
- limited dataset

## Library

- os
- zipfile
- keras
- matplotlib
- tensorflow


## Description

- **Type** : supervised - classification 
  
- **Background** : chess is famous game. it is strategy game and not everyone can become Grandmaster in this field. even the person who have many years experience still learn to increase his ability. in this case, AI have role to surpass human ability so chess can be medium for AI. but the first thing to do is AI must know each items in chess.
  
- **Business Understanding**:
  1. Problem Statement
     - how to make model to classify chess item
   
  2. Goals:
     - to know how to make model to classify chess item

- **Data understanding**
  1. Overview
     - it have 16 chess pieces and it divide based on type namely 1 king (raja), 1 queen (menteri), 2 rook (benteng), 2 knight (kuda), 2 bishop (uskup), 8 pawn (bidak)

     - we use dicoding dataset [link](https://github.com/dicodingacademy/assets/raw/main/ml_pengembangan_academy/Chessman-image-dataset.zip)

     - Label on datasets
  
        | Label | Keterangan |
        | :--: | :-- |
        | 0 | Pawn
        | 1 | King
        | 2 | Knight
        | 2 | Bishop
        | 3 | Rook
        | 4 | Rook
        | 5 | Queen

        urutan menyesuaikan dengan urutan folder ketika diekstrak  
  
     - our dataset have 556 images with pawn (107), king (76), knight (106), Bishop (87), Rook (102), Queen (78)

     -  our dataset have size yang tidak seragam or have ukuran yg berbeda-beda. this is challenging because usually we already get data which have preprocessing and have same size 

  2. Data Preparation
     - do augmentation image to increase dataset item (cause we have limited dataset)
        ```py
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            train_dir = os.path.join('/tmp/Chessman-image-dataset/Chess')
            train_datagen = ImageDataGenerator(rescale=1./255,
                rotation_range=20,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode = 'nearest',
                validation_split=0.1) # set validation split 10% data testin
        ```

     - make data training and data testing
         - our dataset only consist of 1 directory and doesnt split into directory training & testing.
         - but calm down, with ``ImageDataGenerator`` we doesnt need susah payah to split directory manually. cukup dengan ``validation_split``.
         - we only tell dataTraining and dataTesting by add ``subset`` parameter with values ``training`` and ``validation``
        ```py
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(150, 150),
                batch_size=8,
                class_mode='categorical',
                subset='training') # set as training data

            validation_generator = train_datagen.flow_from_directory(
                train_dir, # same directory as training data
                target_size=(150, 150),
                batch_size=16,
                class_mode='categorical',
                subset='validation')
        ```
  3. Modeling
     - ANN Model
     - CNN Architectur (Convulational + MaxPolling)
     - Making MLP (Multi Layer Perceptron) using keras library by calling ``tf.keras.models.Sequential``  
  
       1. Convulational Layer (3) + MaxPoling Layer (decrease resolution) (3)
          - first Convulational is input layer  shape image with 150x150 dimension and 3 bytes color
          - using relu activation func.
          - using Dropout layer 4% to handle overfitting by decrease 4% perceptron each training siklus 
          - using flatten to change CNN result dimention from 2 -> 1
       2. Hidden layer (2)
           - neuron : 512 and 256
           - using activation function : relu
       3. output layer (1)
           - using activation function : softmax
           - return 6 classes, each node contains skor
  
     - model settings :
       - loss function : categorical_crossentropy
       - optimizer : Adam()
       - metrics : accuracy

        ```py
            import tensorflow as tf

            model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.4),  
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.4),  

            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(), 
            
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')  
        ```

## Work Stages

1. download and import datasets
2. extraxt dataset
3. explore dataset (label, jumlah item each label)
4. data preparation
5. build model
6. compile and start train model
7. show graph Loss and Accuray of trained model

## Evaluation

in this cas we got accuration 
- training 0.82   
- testing 0.46 
 

## Code snippet

- dataset augmentation using ``imageDataGenerator``
  - rescale, rotation, zoom, shear
  - fill_mode : nearest
  - validation split : 10% of total data -> data testing
  
  ```py
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
     
    train_dir = os.path.join('/tmp/Chessman-image-dataset/Chess')
    train_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode = 'nearest',
        validation_split=0.1) # set validation split 10% data testing
  ```

- split data into data testing and data training
  ```py
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=8,
        class_mode='categorical',
        subset='training') # set as training data
    validation_generator = train_datagen.flow_from_directory(
        train_dir, # same directory as training data
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='validation')  # set as validation data
  ```
