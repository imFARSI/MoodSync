# MoodSync Video Presentation Script

This document contains the exact spoken script for both presenters. Each person's speech is designed to take approximately 3.5 to 4 minutes when spoken at a normal, clear pace. 

**Instructions for Recording:**
*   Have the `MoodSync.ipynb` notebook open.
*   **Salman** will share his screen first, scrolling through the code as he speaks.
*   When Salman finishes, **Nuha** will take over the screen share (or Salman can continue scrolling for her) to show the baselines, results, and webcam code.
*   *Italicized text in brackets* are instructions for scrolling and pointing. Do not read them aloud!

---

## 🎤 PERSON 1: Salman Farsi (~4 Minutes)
**Topics:** Introduction, Data Preprocessing, Data Augmentation, and FERNET

*(Start with the notebook at the very top)*

"Hello everyone, my name is Salman Farsi, and along with my partner Rifaat Nuha, we present MoodSync. MoodSync is a real-time facial emotion recognition system built using convolutional neural networks, which also features an adaptive music recommendation system based on the user's detected mood. Today, we'll walk you through our Jupyter Notebook implementation.

First, we start with the dataset. We chose the standard FER-2013 dataset, which consists of 48 by 48 pixel grayscale images. For our project scope, we focused on five primary emotions: Angry, Happy, Neutral, Sad, and Surprise. 

*(Scroll down to the Data Preprocessing / Class Balancing code)*

If we scroll down to the data preprocessing section, you'll see how we prepared this data for our neural networks. The original dataset has a significant class imbalance—for instance, there are way more 'Happy' images than 'Surprise' images. To prevent our model from becoming biased towards the majority class, we applied Class Balancing. We randomly sampled exactly 2,000 images for each of the five classes, resulting in a perfectly balanced training set of exactly 10,000 images. 

Next, we normalized the pixel values. By dividing all pixel arrays by 255, we scaled the data between 0 and 1, which helps the neural network converge much faster during training. Finally, as you can see printed here, our training array shape becomes 10,000 images, by 48 by 48 pixels, with 1 color channel for grayscale.

*(Scroll down to the ImageDataGenerator / Augmentation cell)*

Scrolling down to the Data Augmentation block. Even with 10,000 images, deep learning models can easily overfit. To solve this, we used Keras's `ImageDataGenerator`. We applied online augmentation, meaning we randomly rotated the images by 10 to 15 degrees, shifted their width and height by 10 percent, and applied random horizontal flipping. This forces the model to learn actual facial features rather than just memorizing the training set.

*(Scroll down to the FERNET Model Architecture cell)*

Now, let's look at the core of our project: The Model Architectures. I will introduce our main proposed model, which we call FERNET. 

FERNET is a deep, VGG-style Convolutional Neural Network. As we scroll through the code, you can see its architecture. It consists of three main convolutional blocks. Each block has two Conv2D layers using Exponential Linear Unit, or ELU, activations, followed by Batch Normalization, Max Pooling, and Dropout. The filter sizes progressively increase from 64, to 128, to 256, allowing the network to learn highly abstract facial features. In total, FERNET has about 2.33 million trainable parameters.

For training, we used the Adam optimizer. We also implemented two critical callbacks: `EarlyStopping` to halt training if the validation accuracy stopped improving, and `ReduceLROnPlateau`, which dynamically cuts the learning rate in half if the loss plateaus. This allowed us to train the model smoothly up to 50 epochs.

*(Switch screen share from the Jupyter Notebook to the compiled IEEE PDF paper)*

"Before moving to the results, I want to quickly highlight our formal research paper. We documented our entire system into a complete IEEE-formatted conference paper using LaTeX, ensuring our work meets strict academic standards. In this document, we provide a comprehensive literature review, detailed breakdowns of our network architectures, and a rigorous explanation of our training methodology. 

*(Scroll slowly through the PDF to show the graphs and images)*

As you can see, we have also carefully formatted all of our visual evidence. The paper includes the dataset distributions, individual confusion matrices for all three tested models, our final ROC curves, and even live screenshots demonstrating the real-time webcam deployment. This paper serves as the complete academic backbone for the code we are presenting today.

*(Switch screen share back to the Jupyter Notebook or let Nuha take over)*

Now, to prove that FERNET was the best choice, we also built and tested two other baseline models. I'll now hand it over to Nuha, who will explain those baselines, analyze our final results, and show you the real-time webcam demonstration."

---

## 🎤 PERSON 2: Rifaat Nuha (~4 Minutes)
**Topics:** Baseline Models, Results & Evaluation, Real-Time Inference, and Conclusion

*(Nuha takes over screen sharing, or Salman scrolls to the Custom CNN / DenseNet cells)*

"Thank you, Salman. Hello everyone, my name is Rifaat Nuha. As Salman mentioned, to validate FERNET's performance, we first established two baseline models, which you can see here in the notebook.

The first baseline is a Custom CNN. This was designed as a lightweight, 3-block network. In each block, we used standard Conv2D layers with ReLU activation, followed by Batch Normalization, Max Pooling, and a 25% Dropout rate. The filter counts progressed from 32 to 64 to 128. With only about 357,000 parameters, it trained very quickly and converged by epoch 15, but its learning capacity was limited, plateauing at a lower accuracy of around 64.8%.

Our second baseline was DenseNet121, a much deeper transfer-learning architecture. To adapt it for our task, we evaluated it without the pre-trained ImageNet weights and removed its top classification layers. We then appended a Global Average Pooling layer followed by our own Dense classifier with a 50% Dropout rate. Also, because DenseNet expects 3-channel RGB images, we had to duplicate our single grayscale channel to create pseudo-RGB inputs. However, DenseNet121 has over 7 million parameters and its dense connectivity pattern is built for higher resolution images. Because we were training from scratch using only 10,000 low-resolution 48-by-48 pixel images, the network faced severe training instability and only achieved 56% accuracy. 

This proved exactly why FERNET—our custom VGG-style network—was the optimal architecture for this specific dataset. 

*(Scroll down to the Results, Metrics, and Confusion Matrix)*

If we scroll down to the Results and Evaluation section, we can see the final metrics. FERNET achieved the highest accuracy at 71.45%. 

Looking at the Confusion Matrix for FERNET, you can see some interesting biological patterns. The model is exceptionally good at identifying 'Happy' and 'Surprise'—correctly classifying about 82% and 86% of those images, respectively, because smiles and wide eyes are very distinct features. On the other hand, the model occasionally confused 'Sad' and 'Neutral'. This makes sense, as these two emotions share very similar, relaxed facial muscle patterns, especially at a low 48-by-48 pixel resolution.

*(Scroll down to the ROC curve)*

Scrolling to the ROC curves, we calculated the macro-average Area Under the Curve. FERNET achieved an excellent AUC of 0.922, which proves it has high sensitivity and a low false-positive rate across all classes.

*(Scroll down to the Real-Time Webcam Code cell)*

Finally, we come to the most exciting part of the project: the Real-Time Inference Pipeline. Here, we deployed FERNET to work with a live webcam feed. 

In this code block, we use OpenCV’s DNN module with a ResNet-10 SSD face detector to locate faces in the video stream. Once a face is detected, we crop it, convert it to grayscale, resize it to 48 by 48, and feed it to FERNET. 

However, predicting frame-by-frame can cause the text on the screen to flicker rapidly. To fix this, we implemented a 15-frame sliding smoothing buffer. We average the predictions over the last 15 frames using Python's deque, which results in a very stable, smooth output. We also mapped each detected emotion to a music genre—for example, 'Happy' recommends Pop or EDM, and 'Neutral' recommends Ambient Chill music.

*(Scroll to the demo screenshots or show the live webcam output)*

As you can see from our demonstration output, the system successfully detects multiple faces simultaneously, accurately predicts the emotions, and displays the appropriate music recommendation seamlessly in real-time. 

In conclusion, MoodSync successfully demonstrates how optimized deep learning architectures can be deployed for practical, real-time human-computer interaction. Thank you for listening."

---

## 📊 Detailed Project Contribution Breakdown

*(For your own reference or formal submission, here is the exact division of labor showcasing the heavy technical difficulty handled equally by both members).*

### Salman Farsi's Contributions
1. Data Preprocessing: Built the tensor transformations, including dimensional resizing, global pixel normalization, and grayscale conversion.
2. Class Balancing: Mitigated severe dataset skew by algorithmically extracting a uniform distribution of exactly 2,000 images per affective class.
3. FERNET Architecture: Built and trained the primary VGG-style deep convolutional network, integrating Batch Normalization and ELU activations.
4. Training Optimization: Configured dynamic learning rate scheduling (`ReduceLROnPlateau`) and Early Stopping mechanisms to enforce model convergence.
5. Model Evaluation: Computed multi-class Confusion Matrices, categorical accuracy, and weighted F1 metrics to quantify predictive efficacy.
6. Research Paper: Authored the Literature Review, Data Preprocessing, Model Analysis, and Methodology sections adhering to rigorous IEEE formatting.

### Rifaat Nuha's Contributions
1. Baseline Architectures: Formulated the Custom CNN baseline and executed complex dimensionality adaptations for DenseNet121 transfer learning.
2. Data Augmentation: Engineered stochastic online data augmentation (affine rotations, shifts, and flips) to actively suppress overfitting.
3. ROC Curve Analysis: Computed and graphed the macro-average Receiver Operating Characteristic (ROC) curves and associated AUC diagnostics.
4. Real-Time Deployment: Built and optimizedthe end-to-end webcam inference pipeline using OpenCV's ResNet-10 SSD and integrated heuristic logic for music recommendations.
5. Research Paper: Wrote the Abstract, Real-Time Deployment Demonstration, and Conclusion sections of the IEEE report.
