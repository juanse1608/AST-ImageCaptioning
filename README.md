# Image Captioning With Visual Attention
This repository contains our final project about Image Captioning of the Advanced Statistics Topics  course (from the Universidad Nacional de Colombia) made by Juan Sebastián Corredor, Juliana Forero and Jorge Andrés Acosta.

# Table of Contents

1. [Project Goal](#project-goal)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Webapp via Streamlit](#webapp-via-streamlit)
    * [Feedback Storage](#feedback-storage)
    * [Usage](#usage)
4. [References](#references)
## Project Goal
Our goal was to construct a neural network capable of predict or create a caption for a given image. 

We develop the project in Python 3 using Tensorflow.

  <a href="https://www.python.org" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>  <a href="https://www.tensorflow.org" target="_blank"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> 

We use the [MS-COCO](https://cocodataset.org/) dataset as our input in order to train and test the model.

## Neural Network Architecture
We use an encoder-decoder arquitecture where:
* The encoder is a CNN that transform images into features. We use [InceptionV3](https://arxiv.org/pdf/1512.00567v3.pdf) with the weights [image-net](https://www.image-net.org/) as the CNN encoder, in other words we use transfer learning since we don't unfreeze any layer of it.
* The decoder is a RNN (a GRU) that attends to the image (or its features) using a Bahdanau attention mecanism and produces a probality distribution for each token of the vocabulary iteratively.  

If you want to understand better our code you can see all the reading, cleaning and training process in out main notebook: [`Notebook/ImageCaptioning.ipynb`](https://github.com/juanse1608/AST-ImageCaptioning/blob/main/Notebooks/ImageCaptioning.ipynb).

## Webapp via Streamlit

We use the framework (or service) [Streamlit](https://streamlit.io/), a tool that allows to deploy an app in the web using the github. In the web app an user can upload any image, see the predicted caption and sent a feedback about it. For example the user can send a correct caption for the uploaded image if the predicted was wrong. 



### Feedback Storage
<a href="https://cloud.google.com" target="_blank"> <img src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="gcp" width="40" height="40"/> </a>

In order to store the feedback of the users, we use [GCP](cloud.google.com) services: in particular we use BigQuery to store the captions and Storage to store the images.


![Example](https://user-images.githubusercontent.com/46349219/129264122-348a2d4b-dcf7-4601-957d-6ada1140ce0e.gif)


### Usage

You can access to the webapp clicking [here](https://share.streamlit.io/juanse1608/ast-imagecaptioning/main/Scripts/app.py) (if the link is broken is probably beacuse the app is not active, so concact me if you want to try it). 

## References

Some our main references were (that help a lot!):

* [Bahdanau Attention Explained](https://d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html)
* [Attention Models](https://towardsdatascience.com/sequence-2-sequence-model-with-attention-mechanism-9e9ca2a613a)
* [Visual Attention With TF](https://www.tensorflow.org/tutorials/text/image_captioning)
* [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU&ab_channel=CodeEmporium)
* [How to Deploy a Machine Learning Model to Google Cloud for 20% Software Engineers (CS329s tutorial) - YouTube](https://www.youtube.com/watch?v=fw6NMQrYc6w&ab_channel=DanielBourke)
* [Build A Machine Learning Web App From Scratch - YouTube](https://www.youtube.com/watch?v=xl0N7tHiwlw&ab_channel=PythonEngineer)

# 
