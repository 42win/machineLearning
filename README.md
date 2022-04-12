# Machine Learning
Machine learning merupakan bidang studi yang didasari oleh gagasan
bahwa mesin dapat belajar sendiri tanpa diprogram secara eksplisit. Data yang 
digunakan sistem untuk belajar disebut dataset, setiap contoh pelatihan disebut 
trainingset atau sampel. Semakin banyak data, semakin baik pembelajarannya
(Hahn, 2019). Secara umum tipe belajar dari machine learning terbagi menjadi tiga 
metode yaitu Supervised Laerning, Unsupervised Learning, dan Reinforcemen 
Learnin
## ML Workflow

1. Pengumpulan data
2. Exploratory data analysis
3. Data preprocessing
4. Seleksi model
5. Evaluasi model
6. Deployment
7. monitoring

## General step ML Development
1.	Determine purposes 
Ex: i want predict kepadatan  arus lalu lintas in certain day
2.	Make hipotesa 
Ho : weahter likely mententukan kepadatan arus lalu lintas
3.	Collect data
Collect daily data arus lalu lintas and weather
4.	Examine hipotesa 
Train model with collected data
5.	Analisa hasil
Is model predict well?
6.	Make conclusion 
Ternyata weather is not significant to make kepadatan arus lalu lintas
7.	Perbaiki hipotesa and try again
H1: better hipotesis likely hari libur nasional menentukan kepadatan arus lalu lintas

The final goals of ML development is to produce a useful model. The 7th step above kita pandang as experiment that we repeat contiously till get useful model 


## Avoid overfitting
1. Choose model yang lebih sederhana, contohnya in data yang have pola linear use linear regression model than decision tree
2. Decrease data dimension contohnya by using PCA method
3. Add data for pelatihan model jika memungkinkan

## Activation Function ANN
- activation function allow ANN can recognize non-linear pattern
  - ReLu (Rectified Linear Unit) -> ann default function to make network work efficient and speed up computation time
  - Sigmoid -> biner classification
  - tanH
  - Softmax -> more than two classification
