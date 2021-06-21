import numpy as np
from sklearn.svm import SVC
import cv2
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings(action='ignore')
import os
'''
try:
    os.mkdir("dataset/created")
    os.mkdir("dataset/created/train_set")
except:
    pass
import pandas as pd
train = pd.read_csv("dataset/train.csv")
for i in train.Class.unique():
    try:
        os.mkdir("dataset/created/train_set/"+i)
    except:
        pass
import shutil
for i,j in zip(train.Image,train.Class):
    try:
        source = "dataset/train/"+i
          #img = cv2.imread(source)
        destination = "dataset/created/train_set/"+j+"/"+i
        #img = cv2.resize(img, (224, 224,3))
          #cv2.imwrite(destination, img)
        shutil.copy(source, destination)
    except:
        continue
'''
#for i in train.breed.unique():
 # shutil.rmtree("created/train_set/"+i)

def read_training_data(shape1 = 224,shape2 = 224,file="train_set"):
    image_data = []
    target_data = []
    print(os.getcwd())
    for each_letter in os.listdir(file):
        #count = 0
        for each in os.listdir(file+'/'+each_letter):
            #if count < 100:
                path=file+'/'+each_letter
                image_path=os.path.join(path,each)
                img_details = cv2.imread(image_path)
                img_details = cv2.resize(img_details, (shape1, shape2))
                #print(img_details.shape)
                img_details = cv2.cvtColor(img_details,cv2.COLOR_BGR2GRAY)
                #ret2,binary_image = cv2.threshold(img_details,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                flat_bin_image = img_details.reshape(-1)
                image_data.append(flat_bin_image)
                target_data.append(each_letter)
                #count = count+1
            #else:
                #break
    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    accuracy_result = cross_val_score(model, train_data, train_label,cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")
    accuracy_result = accuracy_result * 100
    accuracy_result = ['%.2f' % elem for elem in accuracy_result]
    str1 = ""
    for i in accuracy_result:
        str1 = str1 + i+",       "
    return (str1)

def svc_model(kernel1='rbf',gamma1='scale',max_iter1=- 1, decision_function_shape1='ovr',random_state1=None,shape1 = 224,shape2 = 224,file="train_set"):
    print("\nSVC Model\n")
    print('reading data')
    image_data, target_data = read_training_data(shape1,shape2,file)
    print('reading data completed')
    svc_model = SVC(kernel=kernel1, gamma=gamma1,max_iter=max_iter1, decision_function_shape=decision_function_shape1,random_state=random_state1)
    crossval = cross_validation(svc_model, 5, image_data, target_data)
    print('training model')
    svc_model.fit(image_data, target_data)
    import pickle
    try:
        os.mkdir("models")
    except:
        pass
    print("model trained.saving model..")
    filename = './models/model.pkl'
    pickle.dump(svc_model, open(filename, 'wb'))
    print("model saved")
    return filename,crossval

def adaboost(n_estimators1=100,random_state1=None,learning_rate1=0.1,shape1 = 224,shape2= 224,file="train_set"):
    print("\nADA Boost Model\n")
    print('reading data')
    image_data, target_data = read_training_data(shape1,shape2,file)
    print('reading data completed')
    from sklearn.ensemble import AdaBoostClassifier
    ada_model = AdaBoostClassifier(n_estimators=n_estimators1,random_state=random_state1,learning_rate=learning_rate1)
    crossval = cross_validation(ada_model, 5, image_data, target_data)
    print('training model')
    ada_model.fit(image_data,target_data)
    import pickle
    try:
        os.mkdir("models")
    except:
        pass
    print("model trained.saving model..")
    filename = './models/model.pkl'
    pickle.dump(ada_model, open(filename, 'wb'))
    print("model saved")
    return filename, crossval

def naivebayes(shape1 = 224,shape2= 224,file="train_set"):
    print("\nNaive Bayes model\n")
    print('reading data')
    image_data, target_data = read_training_data(shape1,shape2,file)
    print('reading data completed')
    from sklearn.naive_bayes import GaussianNB
    gnb_model = GaussianNB()
    crossval = cross_validation(gnb_model, 5, image_data, target_data)
    print('training model')
    gnb_model.fit(image_data,target_data)
    import pickle
    print("model trained.saving model..")
    filename = './models/model.pkl'
    pickle.dump(gnb_model, open(filename, 'wb'))
    print("model saved")
    return filename, crossval

def logistic(criterion1='gini', splitter1='best', max_depth1=None, min_samples_split1=2,min_samples_leaf1 =1 ,max_features1=None,random_state1=None,shape1 = 224,shape2= 224,file="train_set"):
    print("\nDecision Tree Model\n")
    print('reading data')
    image_data, target_data = read_training_data(shape1,shape2,file)
    print('reading data completed')
    from sklearn.tree import DecisionTreeClassifier
    dtc_model = DecisionTreeClassifier(criterion=criterion1, splitter=splitter1, max_depth=max_depth1, min_samples_split=min_samples_split1,min_samples_leaf=min_samples_leaf1, max_features=max_features1, random_state=random_state1)
    crossval = cross_validation(dtc_model, 5, image_data, target_data)
    print('training model')
    dtc_model.fit(image_data,target_data)
    import pickle
    try:
        os.mkdir("models")
    except:
        pass
    print("model trained.saving model..")
    filename = './models/model.pkl'
    pickle.dump(dtc_model, open(filename, 'wb'))
    print("model saved")
    return filename, crossval

def knn(weights1 ="uniform",leaf_size1=30,n_neighbors1 = 5,algorithm1 = "auto",shape1 = 224,shape2= 224,file="train_set"):
    from sklearn.neighbors import KNeighborsClassifier
    print("\nKNN model\n")
    print('reading data')
    image_data, target_data = read_training_data(shape1,shape2,file)
    print('reading data completed')
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors1, weights=weights1, algorithm=algorithm1, leaf_size=leaf_size1)
    crossval = cross_validation(knn_model, 5, image_data, target_data)
    print('training model')
    knn_model.fit(image_data,target_data)
    import pickle
    try:
        os.mkdir("models")
    except:
        pass
    print("model trained.saving model..")
    filename = './models/model.pkl'
    pickle.dump(knn_model, open(filename, 'wb'))
    print("model saved")
    return filename, crossval
