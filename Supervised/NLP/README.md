TEST

jangan folder pakai nama algoritma tapi pakai nama case nya nanti di readme di jelaskan al goritma

## List

- fileName
  - binary/multiclass 
  - goals
  - model
  - library 
  - others
  - accuration testing

- [Restaurant Review](https://github.com/42win/machineLearning/tree/main/Supervised/NLP/2_NLP_RestaurantReview)
  - binary
  - classify restaurant reviews are positive or negative 
  - **ANN**  
    - 3 layer (1 hidden layer) 
    - using embedding
    - using GlobalAveragePooling1D
  - pandas, sklearn (train_test_split), tensorflow.keras (preprocessing.text->tokenizer, preprocessing.sequence->pad_sequences), tensorflow, numpy, joblib 
  - 0.82 %

- [Genre Movies](https://github.com/42win/machineLearning/tree/main/Supervised/NLP/3_NLP_GenreMovies)
  - multiclass
  - classify genre movies based on sinopsi 
  - **ANN**  
    - 5 layer (2 hidden layer) 
    - using embedding
    - using LSTM layer
    - using GlobalAveragePooling1D
  - pandas, sklearn (train_test_split), tensorflow.keras (preprocessing.text->tokenizer, preprocessing.sequence->pad_sequences), tensorflow, numpy, joblib 
  - challenge : using LSTM (Long-Short Term Memory) 
  - 0.3 %
