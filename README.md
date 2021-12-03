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

如果要commit `.ipynb` 记得先重新跑一遍， 不然不同cell number 会conflict

#Write down your progress or what you did in the report_draft.txt, so that is will be easier for us to write the report at the end.

#Removed all unrelated images and restored the images path

## Bill：
### In `removeAddtionalImage.ipynb`
  * removed all unrelated images.

### In `getAllFacesInDS1.ipynb`
  * cleaned and filtered csv from dataset1, got all cropped faces using `face_detector` from `facenet` and saved in `./data/cropped.
  * save the dataframe in `data.pkl`

### In `Model_for_race.ipynb`
  * trained a pretrained `vgg2` model from `facenet` on all cropped faces to classificate into 0-other, 1-black, and 2-white 3 classes.
  * saved the model parameters in `dataset1_pic_race_model.pt`
  * Got 0.862 valid accuracy.