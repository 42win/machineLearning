# Classification Restaurant Reviews : Positif or Negatif 

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

- **Type** : supervised - biner classification 
  
- **Background** : at restaurant, costumer reviews are important thing to improve restaurant quality like make analysis marketing, review produk, feed-back produk, service. <br>Nowdays, every companies collect costumer reviews and save it. But some of them dont know how to use it. 
  
- **Business Understanding**:
  1. Problem Statement
     - "The chips and salsa were really good, the salsa was very fresh", is sentence positive or negative reviews?
   
  2. Goals:
     - to make model ML that can classificate costumer reviews into (+) or (-) review

- **Data understanding**
  1. Overview
     - Dataset [link](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set?)
     - it have 1000 rows and 2 column. 
     - dataset column consist of review (text) and label (number 0 or 1)
  
  2. Data Preparation
   
     - split atttribute and label
       ```py
         #atttribute and label
         kalimat = df['sentence'].values
         y = df['label'].values
       ```

     - Train Test Split
       ```py
         # split data training and testing
         kalimat_latih, kalimat_test, y_latih, y_test = train_test_split(kalimat, y, test_size=0.2)
       ```

     - Tokenization, sequences, pad_sequences
        ```py
        # tokenization
        tokenizer = Tokenizer(num_words=250, oov_token='x')
        tokenizer.fit_on_texts(kalimat_latih) 
        tokenizer.fit_on_texts(kalimat_test)

        # change text to sequences
        sekuens_latih = tokenizer.texts_to_sequence(kalimat_latih)
        sekuens_test = tokenizer.texts_to_sequences(kalimat_test)

        # make sequence have same lenght
        padded_latih = pad_sequences(sekuens_latih) 
        padded_test = pad_sequences(sekuens_test)
        ```
        - tokenization : change each word of senteces into numeric
          - num_words : only 250 words yg paling sering muncul yg ditokenization dari 2000 unique words yg ada pada column review
          - oov_token : out of vocabulary, words yg tidak ditokenization akan tetap diberi token 'x'
          - hasil tokenization dapat dilihat saat sequences
   
        - sequences : mengurutkan lagi tokenizated word 
        - pad_sequences : to make all sequences have same length 

   
  3. Modeling
- Model :
  - ANN
  - we make 3 layers
    1. Input Layer
        - using embedding to make ML Model can understand makna each words and group words that have same meaning. Makna of a word didapat dari label
        - parameter embeding : 250 word from tokenization, dimensions we determine by ourself
        - cause our input have 2 dimensions (sequences,label) so we have to convert it to 1 dimension
        - we use GlobalAveragePooling1D to convert it. this function have best performance for NLP than flatten
  
    2. Hidden Layer
        - neuron : 128
        - using activation function : relu

    3. Output Layer
       - using activation function : sigmoid
       - return 1 output

    - Model Settings
      - loss function : binary_crossentropy
      - optimizer : Adam()
      - metrics : accuracy
  
    ```py
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(250, 16) 
    tf.keras.layersGlobalAveragePooling1D(,
    # tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(24activation='relu'),
    tf.keras.layers.Dense(1activation='sigmoid')
    ]) 
    ```

## Work Stages

1. import and load dataset
2. prepare NLP by doing tokenization, sequences, pad_sequences
3. build model using embeding text 
4. compile and start train model
5. evaluate model
6. make prediction using trained model
7. save model

## Evaluation

- in this case we got accuration = 0.82
 

## Code snippet

- show word index
  ```py
  print(tokenizer.word_index)
  ```

- show diagram model
  ```py
    from keras.utils.vis_utils import plot_model

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  ```

- membulatkan angkat keatas
  ```py
  np.round(variabel,jumlahDecimal)
  ```

- test prediction from data test
  ```py
    import numpy as np

    scores = model.predict(padded_test)[5]
    np.round(scores[0],0)
 
  ```

- test prediction from outside data test
  ```py
    # data input
    text = ['The chips and salsa were really good, the salsa was very fresh']

    # preprocessing

    # 1. tokenisation
    sequences = tokenizer.texts_to_sequences(text)

    print('word index')
    print(tokenizer.word_index)

    # 2. change into sequence
    var_txt = tokenizer.texts_to_sequences(text)

    print('\n text disesuaikan dengan index')
    print(var_txt)

    # 3. make input have same lenght
    sequences_samapanjang = pad_sequences(var_txt, 
                                            padding='pre',
                                            maxlen=31)

    print('\n disamakan pangjangnya dengan model')
    print(sequences_samapanjang)

    #do prediction
    scores = model.predict(sequences_samapanjang)[0]

    print('\n hasil prediction')

    print(scores)
    print(np.round(scores,0))

    print('\n sesuaikan dengan label')

    print(y_test[5])
    print(kalimat_test[5])
  ```
