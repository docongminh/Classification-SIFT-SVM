# Classification-SIFT-BoW-SVM
Build a classifier to classsification transport using sift, bag of words and svm

# Tree Project

Root/ 

    data  # Data contains total 1854 image of 5 classes: bus, car, moto, pedestrian 
    
    data_loader.py  # Imread image and label for data
    
    sift_extractors.py # Extract features with SIFT and Build Bag of Word
    
    build_model.py  # Build model SVM and training
    
    gridSearchCV    # implement GridSearchCV find Hyper-params
    
# Hyperparameters
  - Number of centroid in Kmean: 60
  - Penalty parameter C of the error term in SVM: 30

# Label 
  - 'moto': 0
  - 'car': 1
  - 'pedestrian': 2
  - 'truck': 3
  - 'bus': 4
  
# Statistic Data
    - Statistics All Classes
    
 <img src="https://github.com/minhhaui/Classification-SIFT-SVM/blob/master/image/data.png">
    
    - Statistics Classes
    
 <img src="https://github.com/minhhaui/Classification-SIFT-SVM/blob/master/image/statistic.png" >
 
 # Accuracy 
    - Accuracy score: 72.9% - test size: 0.2 
  
 # Using GridSearchCV
    - Best params
    
        'clf__C': 5,
        'clf__gamma': 0.01,
        'clf__kernel': 'rbf'
        
     - Accuracy:  75.72% +/-1.52%
    
 # Requirements
    - opencv-contrib-python==3.4.2.16
    
    - opencv-python==3.4.2.16
    
 ## [More google colab](https://drive.google.com/open?id=1ZyZUT8eAYuyi2Km-tBFC6skSEyG37GTg)
    


    
    
    



