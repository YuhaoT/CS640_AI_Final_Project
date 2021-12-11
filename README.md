# CS640_AI_Final_Project
AI Final Project for BU CS640 
All data put under ./data folder with a ./profile pics folder inside
It would like something similar like the tree below

*__UPDATES__*
- Data for age prediction has uploaded to data/labeled_users_1145s
- new dataset for age have merged to ./data/preprocessed_tweets_with_for_age_pred.csv

```
|-- data/
  |-- labeled_users_1145
     |-- tweets.json
     |-- labeled_users.csv
     |-- profile_pic.zip 
  |-- User demo profiles.json
  |-- Twitter_User_Handles_labeled_tweets.json
  |-- labeled_users.csv
  |-- profile pics/
     |-- 1.png
     |-- 2.png
```

### Presentation Google Slide link: https://docs.google.com/presentation/d/1NyAYcO3dSSaaqkQo6HyI0EygkqZdX5OllcGahSehm7Y/edit?usp=sharing

## Twitter Prediction:
### Logistic Regession in `logisticRegression.ipynb`
use Logistic Regession to get prediction on Race and Age
- use nltk PorterStemmer to stem all words from tweets. 
- Tokenized tweets use sklean TfidfVectorizer with max_features = 5000 and remove English 
- Use 5-fold-cross-validation to get an average score on precision, recall and f1 score

Also tried word2vec vectorizing method with gensim word2vec, but didn't get a better results compared with Logistic regression with Tfidf vectorizer. 

### Bert use (`bert-base-cased`)
we fine-tune bert on race prediction(`bert_race.ipynb`) and age prediction(`bert_age.ipynb`) separately.

- drop nan from data
- tokenize tweets used BertTokenizerFast from Huggingface transformers
- padded all tweets to 25
- concatenate columns 0 ~ 99 columns to one column then sampling train, val, test set from concatenated data
- use one-hot representation to generate label matrix for multi-label representation (for race prediction). 
  - for example [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]] is a one-hot representation for [3, 1, 0]
 - Compute class weights for each label in training set
  - *__Hyperparameters and other tuning__* 
    - use AdamW optimizer
    - Corss Entropy Loss
    - add drop out layer

### Flair prediction on age "Age_flair.ipynb"
Based on bert, we put the data into flair. The accuracy is 0.55 which is lower than 0.56(LR)

## Race Prediction on Profile image:
Created the `race_prediction_image` folder, including all related files. All .ipynb files need to run outside the folder.

### In `removeAdditionalImage.ipynb`
  * removed all unrelated images.

### In `getAllFacesInDS1.ipynb`
  * cleaned and filtered csv from dataset1, got all cropped faces using `face_detector` from `facenet` and saved in `./data/cropped.
  * save the dataframe in `data.pkl`

### In `Model_for_race.ipynb`
  * trained a pretrained `vgg2` model from `facenet` on all cropped faces to classificate into 0-other, 1-black, and 2-white 3 classes.
  * saved the model parameters in `dataset1_pic_race_model.pt`
  * Got BAD result. All predictions are choosing race == white.
  * reduced the number of white labeled pics to make the dataset balanced.
  * Got some result.

### In `CreatPredictRaceCSV.ipynb`
  * Used the model to predict users' race, and saved the result into `userid_race_predrace.csv`.
