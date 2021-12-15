# Traffic sign detection
Intelligent Transportation Systems project.


# Running the code
The final workflow uses models trained during our development process. The main notebook, in which the user can provide the pictures to detect and recognize the traffic signs in it, is [here](scripts/classifying_signs.ipynb). Run all the code following up to the cell with the title "Running the code with pictures", call the function `classify_ROI` (1st param: image, 2nd param: confidence level of SVM sign-or-not binary classification model) on an image of your choice and plot the returned image. 
