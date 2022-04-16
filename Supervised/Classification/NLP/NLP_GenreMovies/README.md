# Genre Movies Classification based on sinopsis

## Challenge

- using LSTM (Long-Short Term Memory) to understand words as berurutan 

## Library

- pandas
- sklearn 
  - train_test_split
- tensorflow.keras
  - preprocessing.text
    - tokenizer
  - preprocessing.sequence
    - pad_sequences 
- tensorflow
- numpy
- joblib 

## Description

- **Type** : supervised - classification multiclass 
  
- **Background** : film is communication media where there is implied message that author want to be  deliver. film have some genre namely romance, horror, thriller, comedy, fantasy, etc. there are a lot of movie lover that is still confused to distinguish movie genre so message of movies cant fully deliver.
  
- **Business Understanding**:
  1. Problem Statement
     - "Anggun (Devi Ivonne) yang mendalami ilmu hitam untuk melampiaskan nafsu setannya selalu mencari korban gadis-gadis cantik untuk sesaji....", what genre of this sinopsis ?
   
  2. Goals:
     - to make model ML that can classificate genre of movie based on sinopsis

- **Data understanding**
  1. Overview
     - Dataset [link](https://www.kaggle.com/antoniuscs/imdb-synopsis-indonesian-movies)
     - it have 1006 rows and 3 column. 
     - dataset column consist of tittle, sinopsis, and genre
     - label column is genre and it have 5 type namely 'Drama','Horor','Komedi','Laga','Romantis'
  
  2. Data Preparation

     - change categorical value into numeric
        ```py
        category = pd.get_dummies(df.genre)
        df_baru = pd.concat([df, category], axis=1)
        ``` 
   
     - split atttribute and label
       ```py
         #atttribute and label
        sinopsis = df_baru['ringkasan_sinopsis'].values
        label = df_baru[['Drama', 'Horor', 'Komedi', 'Laga', 'Romantis']].values
       ```

     - Train Test Split
       ```py
         # split data training and testing
         sinopsis_latih, sinopsis_test, label_latih, label_test = train_test_split(sinopsis, label, test_size=0.2)
       ```
    
     - Cek number of unique words
        ```py
        len(tokenizer.word_index) 
        ```

     - Tokenization, sequences, pad_sequences
        ```py
        # tokenization
        tokenizer = Tokenizer(num_words=5000, oov_token='x')
        tokenizer.fit_on_texts(sinopsis_latih) 
        tokenizer.fit_on_texts(sinopsis_test)
        
        # change text to sequences
        sekuens_latih = tokenizer.texts_to_sequences(sinopsis_latih)
        sekuens_test = tokenizer.texts_to_sequences(sinopsis_test)
        
        # make sequence have same lenght
        padded_latih = pad_sequences(sekuens_latih) 
        padded_test = pad_sequences(sekuens_test)
        ```
        - tokenization : change each word of senteces into numeric
          - num_words : only 5000 words yg paling sering muncul yg ditokenization dari 13678 unique words yg ada pada column review
          - oov_token : out of vocabulary, words yg tidak ditokenization akan tetap diberi token 'x'
          - hasil tokenization dapat dilihat saat sequences
   
        - sequences : mengurutkan lagi tokenizated word 
        - pad_sequences : to make all sequences have same length 

   
  3. Modeling
- Model :
  - ANN
  - we make 4 layers
    1. Input Layer
        - using embedding to make ML Model can understand makna each words and group words that have same meaning. Makna of a word didapat dari label
        - parameter embeding : 5000 word from tokenization, dimensions we determine by ourself 
        - using LSTM to make ML Model can understand makna each words secara berurutan
  
    2. Hidden Layer (2)
        - neuron : 128 and 64
        - using activation function : relu

    3. Output Layer
       - using activation function : softmax
       - return 5 output

    - Model Settings
      - loss function : categorical_crossentropy
      - optimizer : Adam()
      - metrics : accuracy
  
    ```py
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.LSTM(64), 

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(5, activation='softmax')
    ])
    ```

## Work Stages

1. import and load dataset
2. prepare NLP by doing categorical values change, tokenization, sequences, pad_sequences
3. build model using embeding text 
4. compile and start train model
5. evaluate model
6. make prediction using trained model
7. save model

## Evaluation

- in this case we got accuration = 0.3
 

## Code snippet

- show word index
  ```py
  print(tokenizer.word_index)
  ``` 

- test prediction from data test
  ```py
    import numpy as np

    #define labels
    labels = ['Drama','Horor','Komedi','Laga','Romantis']

    test_y = label_latih[5] 

    print( labels[np.argmax(test_y)] )
 
  ```

- test prediction from outside data test
  ```py
    
    labels = ['Drama','Horor','Komedi','Laga','Romantis'] #define labels

    # data input
    text = ['Anggun (Devi Ivonne) yang mendalami ilmu hitam untuk melampiaskan nafsu setannya selalu mencari korban gadis-gadis cantik untuk sesaji. Ayu (Hazni Zulaikah H) yang hendak menyelamatkannya dari dunia hitam malah hampir menjadi korban. Untung hadir Iman (Harry Capri) sebuah sosok gaib tapi manusia yang membantu Ayu. Anggun akhirnya mati karena ledakan yang datang dari langit.']

    # preprocessing 
    sequences = tokenizer.texts_to_sequences(text) # 1. tokenisation 
    var_txt = tokenizer.texts_to_sequences(text) # 2. change into sequence  
    sequences_samapanjang = pad_sequences(var_txt) # 3. make input have same lenght  

    #do prediction
    scores = model.predict(sequences_samapanjang)[0]
    print( labels[np.argmax(scores)] )
  ```
