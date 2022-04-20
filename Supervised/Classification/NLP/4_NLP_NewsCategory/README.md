# News Category Classification

# Challenge
- dataset minimal 1000 sample
- using LSTM in model architect
- using sequential model
- validation set 20%
- using embeding
- using tokenizer
- accuration minimal 75%
- implement call back
- make plot loss and accuration 

# Library
- pandas
- nltik
  - nltk.corpus (stopword) 
- os, re, string
- keras
  - keras.layers (Input, LSTM, Bidirectional, SpatialDropout1D, Dropout, Flatten, Dense, Embedding, BatchNormalization)
  - keras.models (Model)
  - keras.callbacks (EarlyStopping)
  - keras.preprocessing.text (Tokenizer, text_to_word_sequence)
  - keras.preprocessing.sequence (pad_sequence)
  - tensorflow.keras.utils (to_categorical)

- **Bussiness Understanding**:
  1. Problem Statement
     - Technology is important part of society life. people use it to share or get information. one of that information can be news. in general news are delivered in website and consist of some category like politic, sport, economic, health, etc. so far, news grouping into certain category is done by editor manually. to do it editor have to know entire news content but if there are a lot of news article and varieted category it will be difficult especially if each category have content similarity. this case need accuracy and much time. so we need to make system to solve this case.
  2. Goals
      - To make model ML that can classificate category of news based on tittle and content text
  
- **Data Understanding**
  1. Overview
     - Dataset [link](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive) 
     - it have 2225 rows and 4 columns
     - dataset columns consist of category, filename, title, content
     - label column is category and it consist of bussiness, entertaiment, politics, sport, and tech
  
  2. Data Preparation

     - make all char lower-case
       ```py 
         df_new.title = df_new.title.apply(lambda x: x.lower())
         df_new.content = df_new.content.apply(lambda x: x.lower())
       ```

     - remove punctuation (tanda baca)
       ```py
       def cleaner(data):
         return(data.translate(str.maketrans('','', string.punctuation)))
         df_new.title = df_new.title.apply(lambda x: cleaner(x))
         df_new.content = df_new.content.apply(lambda x: lem(x))
       ```

     - Lemmatization (NLP Techniq to return word to kata dasarnya)
       ```py
       import nltk
       from nltk.stem import WordNetLemmatizer

       lemmatizer = WordNetLemmatizer()
       def lem(data):
         pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
         return(' '.join([lemmatizer.lemmatize(w,pos_dict.get(t, wn.NOUN)) for w,t in nltk.pos_tag(data.split())]))
         df_new.title = df_new.title.apply(lambda x: lem(x))
         df_new.content = df_new.content.apply(lambda x: lem(x))
       ```

     - removing number
       ```py
       def rem_numbers(data):
         return re.sub('[0-9]+','',data)
         df_new['title'].apply(rem_numbers)
         df_new['content'].apply(rem_numbers)
       ```

     - removing stop words (general words that have not meaning like the,is,of in english)
       ```py
         import nltk
         nltk.download('stopwords')

         st_words = stopwords.words()
         def stopword(data):
             return(' '.join([w for w in data.split() if w not in st_words ]))
             df_new.title = df_new.title.apply(lambda x: stopword(x))
             df_new.content = df_new.content.apply(lambda x: lem(x)
       ```

     - change categorical values into numeric
       ```py
       # data category one-hot-encoding
         category = pd.get_dummies(df_new.category)
         df_new_cat = pd.concat([df_new, category], axis=1)
         df_new_cat = df_new_cat.drop(columns='category')
       ```

     - Seperate label and column
       ```py
       # change dataframe value to numpy array
         news = df_new_cat['title'].values + '' + df_new_cat['content'].values 

         label = df_new_cat.values[:, 1:].astype(float) 
       ```

     - Tokenization, sequences, pad_sequences
       ```py
         # tokenizer
         from tensorflow.keras.preprocessing.text import Tokenizer
         from tensorflow.keras.preprocessing.sequence import pad_sequences
         
         tokenizer = Tokenizer(num_words=5000, oov_token='x', filters='!"#$%&()*+,-./:;<=>@[\]^_`{|}~ ')
         tokenizer.fit_on_texts(news_train) 
       
         sekuens_train = tokenizer.texts_to_sequences(news_train)
         sekuens_test = tokenizer.texts_to_sequences(news_test)
         
         padded_train = pad_sequences(sekuens_train) 
         padded_test = pad_sequences(sekuens_test)
       ```

  3. Modeling

     - ANN
     - we make 5 layer
       - input 
         - using embedding layer  (embeding dimensi input : 5000 -> output dimensi : 64)
         - using LSTM (Long-Short Term Memory) to make model understand words sequentially with neuron 128
       - hidden 
         - neuron : 128 using relu activation
         - using Dropout layer to avoid overfitting
       - output
         - return 5 output
         - using softmax activation function

     - model settings
       - loss function : categorical_crossentropy
       - optimizer : Adam()
       - metrics : accuracy

        ```py
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
                tf.keras.layers.LSTM(128),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(5, activation='softmax')
            ])
        ```

## Work Stages

1. import and load dataset
2. Data Exploration (show data,column name, number of rows and column)
3. Data Preprocessing
   - all char lowercase, remove functuation, lematization, removing number, removing stopword
   - change categorical to numeric
   - seperate label and attribute
   - split dataset and datatraining     
4. prepare NLP by doing tokenization, sequences, pad_sequences
5. build model using embeding, LSTM Layer
6. train model
7. evaluate model
8. Save model
9. make prediction using trained model 

## Evaluation

- in this case we got accuration = 0.9