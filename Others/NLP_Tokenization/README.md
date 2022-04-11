# Tokenization
- Proses mengonversi kata-kata ke dalam bilangan numerik 

## Library

- tensorflow
  - Tokenizer
  - Pad_sequences
 
## Description

- **Type** : -
  
- **Background** : Make ML Model can imitate Human ability to recognize sentences in every languanges in the world
  
- **Business Understanding**:
  1. Problem Statement
     - how ML Model can process sentences
   
  2. Goals:
     - to know how ML Model process sentences

- **Data understanding**
  1. Overview
      - it just 3 dummy sentences
  

## Work Stages

1. Import Library
2. Make object Tokenizer
3. Make text dummy
4. do tokenization to dataset and change to sequence
5. do padding to make it same length

 
## Code snippet

- meke tokenizer object
  ```py
  tokenizer = Tokenizer(num_words= 15, oov_token='-')
  ```
  - num_words : jumlah kata yg akan dikonversi into token/numberic
  - oov_token : parameter yg berfungsi to change words yg tidak ditokenisasi

- do tokenization
  ```py
  tokenizer.fit_on_texts(teks)
  ```

- change tokenized text into sequence
  ```py
  sequences = tokenizer.texts_to_sequences(teks)
  ``` 

- show pair tokenization and words
  ```py
  print(tokenizer.word_index)
  ```

- change each sentences into suitable token
  ```py
  print(tokenizer.texts_to_sequences(['Saya suka belajar programing sejak SMP']))
  ```

- do padding
  ```py
   from tensorflow.keras.preprocessing.sequence import pad_sequences
    sequences_samapanjang = pad_sequences(sequences)
    
    print(sequences_samapanjang)
  ```

- to adjust pading result
  ```py
      sequences_samapanjang = pad_sequences(sequences, 
                                          padding='post',
                                          maxlen=5,
                                          truncating='post')
  ```
  - padding='post' : values align left
  - maxlen : to limit length (get values from right)
  - truncating='psot' : maxlen get values from left
