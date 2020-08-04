# Real or Not? - Kaggle NLP Binary Classification

## Background
In this project, we will evaluate two models that predict whether a given tweet is about a real disaster or not - SVM Classifier and LSTM Neural Network. In order to achieve the objective, there are several sections as below:

0. Data Preprocessing;
1. SVM Classifier;
2. LSTM Model;
3. Conclusion.

## Data Preprocessing

Run the following command to download the Kaggle data:
```
kaggle competitions download -c nlp-getting-started
```

In this repo, the downloaded data is already placed in folder [/data](./data). The details of columns are following:

* id - a unique identifier for each tweet
* text - the text of the tweet
* location - the location the tweet was sent from (may be blank)
* keyword - a particular keyword from the tweet (may be blank)
* target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

Here, we simply use `text` column as input to generate linguistic features. Two approachs are applied to do feature extraction: TF-IDF and GloVe word embeddings. `target` column is used as the target variable. 

### Setups before model comparison

* All the data in `train.csv` are split into 80/20 percent as training/testing data (`test.csv` and `sample_submission.csv` are not used).
* Training data will be used to run a grid search process (using 3-fold cross validation) for finding the optimal parameters for SVM Classifier (no grid search in LSTM model); while testing data is used to get the validation score for model comparison. 
* random_state' is fixed in data-splitting as 3 in order to get consistent results.


## SVM Classifier
The SVM Classifier model is trained and evaluated in [Jupyter notebook](./notebooks/1.1-diaster_tweets_TFIDF_SVM.ipynb). 

Using TF-IDF for feature extraction, we set `min_df = 0.01` to reduce the size of the vocabulary from 21637 to 150. This will help us remove the rare vocabularies in model training. 

The model is trained on a grid of parameters: `parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100]}`. After grid search, the best model is `C=1.0, kernel='rbf'`. Details of the accuracy and classification report on the test data are as below:

```
Accuracy: 0.7399.

Classification report:

                    precision    recall  f1-score   support

                0       0.71      0.90      0.79       841
                1       0.81      0.55      0.65       682
         accuracy                           0.74      1523
        macro avg       0.76      0.72      0.72      1523
     weighted avg       0.75      0.74      0.73      1523
```

## LSTM Model
In this section, we will look into the LSTM model. This model is trained and evaluated in [Jupyter notebook](./notebooks/1.2-diaster_tweets_GloVe_LSTM_keras.ipynb). We will use the following layer architecture for the neural network:
```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 25)                0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 25, 50)            59675750  
_________________________________________________________________
lstm_1 (LSTM)                (None, 25, 64)            29440     
_________________________________________________________________
dropout_1 (Dropout)          (None, 25, 64)            0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 64)                33024     
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 130       
_________________________________________________________________
activation_1 (Activation)    (None, 2)                 0         
=================================================================
Total params: 59,738,344
Trainable params: 62,594
Non-trainable params: 59,675,750
_______________________________________
```

### Quick explanation of the embedding layer
Pre-trained (twitter) 50-dimensional GloVe embeddings is used. You can find the GloVe model in [/models](./models). We  will initialize the Embedding layer with the GloVe 50-dimensional vectors and set the layer to be un-trainable. That is, we will leave the GloVe embeddings fixed as pre-trained vectors. 

Each sentence from the training batch will convert to a list of word indices, before putting into embedding layer. The output is an array of `(None, 25, 50)` which will be feed into LSTM layer. Note that, the maximum length of input sentence is limited to 25 in order to avoid zero-sparsed data.

When training the neural network, `epochs` is set to 50 and `batch_size` is 30. This gives us the following results:

```
Accuracy: 0.7938.

Classification report:

                    precision    recall  f1-score   support

                0       0.78      0.88      0.82       841
                1       0.82      0.69      0.75       682

         accuracy                           0.79      1523
        macro avg       0.80      0.78      0.79      1523
     weighted avg       0.80      0.79      0.79      1523
```

## Conclusion
Based on these requirements, we've got the results as table below:

optimal model | test accuracy 
--- | --- 
SVC(C=1, kernel='rbf') | 0.7399  
LSTM_Model | 0.7938

From the table we can see, we achieved the highest accuracy of 0.7938 from LSTM model. 

We could of course spend more time tuning these two models, in order to achieve a better accuracy score. For example, we can experiment more thresholds when generating TF-IDF vectors, or adding different layers than LSTM in the neural network. But also, we can involve more models into the comparison like Logistic Regression, adapting BERT language model into the neural network.

