#   Finger count image classification
File explanation:
-   classification_from_dir.py - takes two parameters: source and target directory. Images of hands from the source directory are classified and copied to target directories sub directories.
-   fingers.zip - conatins my own created dataset I used for the training of the model.
-   train_finger_count.ipynb - is the file used for the training of model.
-   utils/classification.py - contains function to be used to preprocess images to be used with the model.

#   Finger count model
The model achieved 99% training accuracy and 95% validation accuracy. The model works best on solid background. The model works surprisingly well when counting any finger combination, even those it was not trained on.