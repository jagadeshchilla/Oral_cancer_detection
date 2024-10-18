# 🦷 Oral Cancer Detection


<img src="./assets/lottie.gif" width="600" height="400" alt="GIF Demo">

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-Enabled-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red.svg)
![Lottie Files](https://img.shields.io/badge/Lottie_Files-Enabled-brightgreen.svg)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Enabled-blue.svg)
![OpenCV2](https://img.shields.io/badge/OpenCV2-Enabled-green.svg)
![Numpy](https://img.shields.io/badge/Numpy-Enabled-yellow.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV-Python-Headless](https://img.shields.io/badge/OpenCV--Python--Headless-Enabled-blue.svg)
![Pillow](https://img.shields.io/badge/Pillow-Enabled-purple.svg)
![Streamlit-Option-Menu](https://img.shields.io/badge/Streamlit--Option--Menu-Enabled-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-Enabled-darkblue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Enabled-lightblue.svg)
![Streamlit-Lottie](https://img.shields.io/badge/Streamlit--Lottie-Enabled-pink.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Enabled-green.svg)
![Graphviz](https://img.shields.io/badge/Graphviz-v0.20.3-lightgray.svg)
![Pydot](https://img.shields.io/badge/Pydot-Enabled-blue.svg)
![Gdown](https://img.shields.io/badge/Gdown-5.2.0-orange.svg)
![LibGL1](https://img.shields.io/badge/LibGL1-Enabled-darkgreen.svg)
![GitHub](https://img.shields.io/badge/GitHub-Enabled-black.svg)


## Overview

The **Oral Cancer Detection** project aims to develop a robust machine learning model that leverages advanced image processing techniques to accurately identify signs of oral cancer in images. This project integrates various technologies and frameworks, providing a seamless user experience for both medical professionals and patients.

## Key Features

- **📈 Accurate Detection:** Utilizes state-of-the-art deep learning algorithms for precise identification of oral cancer.
- **🖼️ Image Processing:** Implements OpenCV for enhanced image preprocessing, ensuring high-quality input for the model.
- **📊 User-Friendly Interface:** Built with Streamlit to offer an intuitive interface for users to upload images and view results.
- **🚀 Fast Performance:** Optimized for quick processing and real-time feedback.
- **📊 Data Visualization:** Includes interactive visualizations using Matplotlib and Graphviz for better understanding of model predictions.
- **🔄 Model Training and Evaluation:** Supports training with various datasets and evaluating model performance with scikit-learn.
- **☁️ Cloud Deployment:** Enables deployment using Kubernetes for scalability and reliability.

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Solution](#solution)
4. [Dataset](#dataset)
5. [Methodologies](#methodologies)
6. [Models Used](#models-used)
7. [Model Comparisons](#model-comparisons)
8. [Building Interface](#building-interface)
9. [Deployment](#deployment)
10. [Results](#results)
11. [Conclusion](#conclusion)
12. [Installation](#installation)
13. [Future Works](#future-works)
14. [References](#references)


## Introduction

The **Oral Cancer Detection** project aims to harness the power of machine learning and image processing to accurately detect oral cancer at an early stage. 🦷 Oral cancer is a significant health concern, often leading to severe consequences if not identified promptly. This project seeks to provide a reliable tool that aids medical professionals in diagnosing oral cancer through advanced techniques. 🩺

Utilizing a comprehensive dataset of oral images, this project implements various algorithms to train models capable of distinguishing between healthy and cancerous tissues. 📊 The application features a user-friendly interface, enabling healthcare practitioners to upload images and receive instant feedback on potential cancer detection. With the integration of visualization tools, users can gain insights into the model's predictions and the underlying data. 🔍

By leveraging state-of-the-art technologies, this project not only aims to improve diagnostic accuracy but also to facilitate early intervention, ultimately contributing to better patient outcomes. 🌈

## Problem Statement

Oral cancer is a major global health issue, accounting for hundreds of thousands of cases annually. Despite medical advancements, early detection of oral cancer remains a challenge, often leading to late-stage diagnoses and poor patient outcomes. 🦷 The lack of accessible and reliable diagnostic tools, particularly in remote and underserved areas, exacerbates this problem.

The need for a solution that allows medical professionals to identify oral cancer at an early stage is critical. 📉 Early detection can significantly improve survival rates and reduce treatment costs. Therefore, this project focuses on building a machine learning-based tool that assists in the early detection of oral cancer through image analysis, addressing both accessibility and diagnostic accuracy. 📲

## Solution

The **Oral Cancer Detection** project provides a machine learning-based solution to assist healthcare professionals in detecting oral cancer early. 🦷 By utilizing advanced deep learning techniques, this project processes and analyzes medical images to identify potential cancerous regions in the oral cavity. 

This solution includes the following key components:

- **📸 Image Processing:** Uses OpenCV to preprocess images, enhancing the clarity and quality of input data for more accurate detection.
- **🤖 Machine Learning Models:** Employs cutting-edge deep learning models built using TensorFlow to classify images into cancerous and non-cancerous categories.
- **🖥️ User-Friendly Interface:** Features an easy-to-use Streamlit-based interface, enabling users to upload images and receive instant diagnostic results with high accuracy.
- **📊 Data Visualization:** Visualizes the prediction results and the areas of interest within the image, making it easier for healthcare professionals to interpret the results.
- **☁️ Cloud-Ready Deployment:** The model is scalable and can be deployed using Kubernetes for real-time and widespread use in clinical settings.

This approach not only makes cancer detection faster and more accessible but also enhances diagnostic precision, leading to better patient outcomes and earlier interventions. 🌟

## Dataset

### About the Dataset

Introducing the **Oral Cancer Image Dataset**! This dataset comprises 500 oral cancer images and 450 non-cancer oral images, all meticulously labeled for seamless classification. 🦷 The dataset is designed to support research and development in the field of oral cancer detection using advanced machine learning algorithms.

With a balanced representation of cancer and non-cancer samples, it allows researchers to explore innovative approaches to enhance diagnostic accuracy. 🔬 This dataset serves as a valuable resource for the healthcare community, fostering advancements in early detection and intervention for oral cancer. 💡

You can access the dataset [here](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset) 📂.

## Methodologies

<details>
  <summary>📊 Click to view the Flowchart</summary>
  
  ![Flowchart](./assets/flow.jpg)
  
</details>

### Process Overview

1. **📥 Data Collection:**
   Medical images are sourced from reliable and reputable datasets, ensuring a comprehensive mix of oral cancer and non-cancer samples. This provides a strong foundation for training the model, ensuring that it learns from high-quality, representative data.

2. **🛠️ Preprocessing:**
   To ensure consistency, all images are resized to a standard dimension of 260x260 pixels with 3 color channels (RGB). The images are normalized to a range between [0,1] for smoother training. Image augmentation techniques, such as rotation and flipping, are applied to make the model robust against variations and prevent overfitting.

3. **🧠 Model Selection:**
   A range of cutting-edge deep learning architectures are chosen for comparison:
   - **CNN (Convolutional Neural Networks):** A standard deep learning model for image classification.
   - **ResNet50:** A deeper network that addresses the vanishing gradient problem using skip connections.
   - **DenseNet121:** A model that efficiently passes gradients between layers using dense connections.
   - **EfficientNetB2:** A state-of-the-art model that balances accuracy and efficiency through compound scaling.
   - **VGG19:** A popular deep learning model with a simple, uniform architecture known for its performance in image tasks.

4. **🎓 Training:**
   Each model is trained using the preprocessed images. During training, the models adjust their weights using a process called **backpropagation** to minimize the loss function. The training continues for several epochs until the models converge, or achieve optimal performance on the training data.

5. **📊 Evaluation:**
   After training, the models are evaluated based on:
   - **Accuracy:** The proportion of correctly predicted labels.
   - **Speed:** How quickly the model processes new data.
   - **Memory usage:** The amount of system resources required by the model.
   Performance metrics such as **precision**, **recall**, and **F1 score** are calculated to assess how well the models balance true positives and false negatives.

6. **📈 Comparison:**
   Once all models are trained and evaluated, their performances are compared. The model that strikes the best balance between accuracy, speed, and resource efficiency is selected for deployment. This ensures that the deployed model is optimal for real-world use.
