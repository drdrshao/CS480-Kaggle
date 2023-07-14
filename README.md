# CS480-Kaggle
## 1. Data Pre-processing
- We apply resize and random vertical flip on the original dataset. The resize is applied to increase the training time and the vertical flip is used to accommodate the situations that some pictures in the training set are vertically flipped. Afterwards, we applied normalization on the training dataset.

## 2. Model Choices
- We used the pretrained ResNet50 Model as follows:
  model = models.resnet50(pretrained=True).to(device)
- Then we used one layer of linear fully-connected layer to accommodate our classification purpose:
  model.fc = nn.Sequential( nn.Linear(2048, 22)).to(device)
We used 22 because we have 22 labels.
- We chose ResNet50 due to its high precision.

## 3. Training
- We used Adam as the optimizer because it could automatically adapt its learning rate and it is relatively easy to set up. Moreover, it generally produces better results.
- We used CrossEntropyLoss as the criterion. This criterion computes the cross-entropy loss between input and target. It is useful when training a classification problem with 'C' classes.
- We found that increasing the epoch amount would have a better result, therefore, we chose 140 epochs because after 140 epochs, there was no significant improvement on the loss.

## 4. Predicting
- We transform the test data by applying resize and normalization. Then, we feed the transformed test data into the trained model. We applied the softmax function on the results to get the probabilities that the model predicts the picture to be a specific class.
- We used the class that has the greatest probability as the prediction result.

## 5. Note
- Since we have only 5 attempts each day, therefore we made testing labels just for testing results. We never used the testing labels and any information in the testing dataset for training and tuning parameters.
