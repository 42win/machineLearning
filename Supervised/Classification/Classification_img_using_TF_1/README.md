# Image Classification 

using CNN (Concvulational Neural Network)
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
