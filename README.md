## Lung Cancer Detection using Deep Learning

Buidling a package that detect the lung cancer with the help of images using cnn/ transfer learning
To run the project do the necessary steps :

*constraints*

python 3.8 version

*set run environment*

clone the repository using 
```
git clone https://github.com/ups1610/lungs_cancer_detection-.git
```
create virtual env using
```
conda create -p env_name python==3.8
cd lungs_data
```
install all the libraries
```
pip install -r requirements.txt
```

### Dataset Preparation

dataset can be collected from kaggle https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images. 
It contains 5k images in jpeg format.
Download the zip file from the kaggle and store the zip folder under the folder where other files settled down using the command

```
python dataset_import.py
```


### Data visualization

After download the dataset run the command
```
python data_visualization.py
```
This file created to get the visualization of data i.e images in form of jpeg 

### Data Preprocessing

run the command

```
python pre_processing.py
```

This file is used to convert the image in array form containg the pixel values .

For that data is then returning images pixels i.e X and performed one hot encoded values.

### Model Preparing, Creation and Evaluation

run the command 

```
python model.py
```

Data then divided in training phase and testing phase.

Using the CNN model data get trained and a model is stored in history variable containing the summary of the model .

Before training the model model get hyper-tuined to increase the accuracy and burn the less loss.

After the creation of the model data get visualized using graphs showing the performance of the data.

Further confusion matrix is created to get the whole details of the data with an accuracy of more than 91%