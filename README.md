# Speech Recognition/Classification using Deep Neural Networks

While todays interaction between man and machine is done many through screens a much more
natural way to do so is via speech. Many industry leaders are already developing and improving
algorithms that are meant to mimic and understand human language. Achieving human like
performance will open a completely new way to interact with devices and offer new applications we
haven’t thought of before. In this Project we test state of the art methods in machine learning using
deep neural networks for “understanding” words and develop new architectures that try to capture
all the features of human language.
####Keywords: Deep Neural Networks, Speech Recognition, Speech Classification 

##Model
We aim to build a Deep neural network that will be able to distinguish between 12 classes: 'Yes',
'No', 'Up', 'Down', 'Left', 'Right', 'On', 'Off', 'Stop', 'Go', Silence and Unknown. The input data to the
net are 1 second recordings of different people saying a word. Below are schematics of the Networks
tested.
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Kaglle-Tensorflow-Speech-Recognition/master/Images/model_image.PNG" width="400" title="model_2">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Kaglle-Tensorflow-Speech-Recognition/master/Images/model_image_2.PNG" width="400" title="model_2">
</p>

##Results
The Networks where tested on the Kaggle TensorFlow Speech Recognition Challenge – data set 
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Kaglle-Tensorflow-Speech-Recognition/master/Images/precision_recall.PNG" width="400" title="precision_recall_1">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Kaglle-Tensorflow-Speech-Recognition/master/Images/precision_recall_2.PNG" width="400" title="precision_recall_2">
</p>

Results on the Speech recognition data set Test set
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/Kaglle-Tensorflow-Speech-Recognition/master/Images/result_table.PNG" width="400" title="Results_table">
</p>

## Authors

* **Etai Wagner** 
* **Eyal Ben David** 
