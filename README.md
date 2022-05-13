# Captioning-on-Sport-Images

### Project Description
Image captioning is the automatic generation of the textual description of an image. In our project, sport related images are captioned.

### Dataset
- Sport related images are acquired from both existing dataset,[flickr 30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and manual collection from google.
- There are at least four different captions for each image of 12 sport types. 
- 70% (1135) of dataset is used as training and 30% (480) is used as testing data. 

### Methodology
- Preprocessing: resize all images into 100 x 100, vectorize the captions
- Encoder (Feature Extraction): ResNet50
- Decoder (Sequence Generation): LSTM
- Evaluation: Bilingual Evaluation Understudy(BLEU) score
