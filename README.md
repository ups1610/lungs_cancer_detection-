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
Download the zip file from the kaggle and store the zip folder under the folder where other files settled down.

### Data visualization

After download the dataset run the command
```
python data_visualization.py
```