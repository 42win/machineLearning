# Image Classification 
- classify tidy room and messy room

framework : TensorFlow

library : Keras
- API to deploy Artificial neural network
- Allow us to make a multi layer perceptron and convolutional neural network easly

architecture : CNN (Concvulational Neural Network)
-	Arsitektur CNN is a neural network yang using a layer convulational and max pooling

## Tujuan
the first step before we build model ML is define problem statement that we want to selesaikan.
- what problem we want to overcome and how we implement model Artificial Neural Network
- in this case we want to make a model to clasify image of room and predict either the room is tidy or untidy 


## Tahapan Training
for the first step we have to understand our dataset.
- format data, jumlah sample, jumlah label.
- ensure out dataset is data kontinu (regression problem) or data diskrit (classification problem)
- our dataset have 200 sample data training that consist of 100 sample tidy room image and 100 sample untidy room image

- training stages 
   1. ensure TF(TensorFlow) is above 2.0 version 
   2. donwload dataset and extract file by unzip method
   3. menampung directory each class in train directory and validation into varaiabel
   4.  Pre-processing data dengan image augmentation
   5.  build model architect using Convolutionl Neural Network (CNN)
   6.  compile and train model using ``model.compile`` dan ``model.fit`` till get accuracy that we want
   7.  examine model that have been made using image yg belum dikenalai model


- install tensorflow di anaconda [link tut](https://www.youtube.com/watch?v=otzZRZtXlOs)


## Code

- import tensorFlow ``import tensorflow as tf``
- ekstak zip file
  ```py
   import zipfile,os

    local_zip = '/tmp/messy_vs_clean_room.zip'

    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp')
    zip_ref.close()
  ```
- define path
  ```py
  base_dir = '/tmp/images'
  train_dir = os.path.join(base_dir, 'train')validation_dir = os.path.join(base_dir, 'val')
  ```

- imageDataGenerator
  ```py
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
     
   train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=20,
                        horizontal_flip=True,
                        shear_range = 0.2,
                        fill_mode = 'nearest')
  ```

- prepare dataTraining and dataTesting from file directory
  ```py
  train_generator = train_datagen.flow_from_directory(
            train_dir,  # direktori data latih
            target_size=(150, 150),  # mengubah resolusi seluruh gambar menjadi 150x150 piksel
            batch_size=4,
            # karena ini merupakan masalah klasifikasi 2 kelas, gunakan class_mode = 'binary'
            class_mode='binary')
  ```

- build CNN Architect Model
  ```py
  model = tf.keras.models.Sequential([
        
        #input
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

        #hidden layer
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),

        #output
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
  ```

- compile model. to make model learn
  ```py
     model.compile(loss='binary_crossentropy',
                  optimizer=tf.optimizers.Adam(),
                  metrics=['accuracy'])
  ```
  optimizer and loss function disesuaikan dengan klasifikasi biner or multiple

- implement model to data
  ```py 
    model.fit(
          train_generator,
          steps_per_epoch=25,  # berapa batch yang akan dieksekusi pada setiap epoch
          epochs=20, # tambahkan epochs jika akurasi model belum optimal
          validation_data=validation_generator, # menampilkan akurasi pengujian data validasi
          validation_steps=5,  # berapa batch yang akan dieksekusi pada setiap epoch
          verbose=2)
  ```
