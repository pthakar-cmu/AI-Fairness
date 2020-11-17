from sklearn.datasets import make_classification 
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
from collections import defaultdict
from xgboost import XGBClassifier
import matplotlib.pyplot as plt 
from pprint import pprint 
import xgboost as xgb
import numpy as np 
import pandas as pd
import seaborn 
import warnings

np.random.seed(sum(map(ord, "aesthetics")))
seaborn.set_context('notebook') 
seaborn.set_style(style='darkgrid')


def get_eval1(clf, X,y):
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(clf, X, y, cv=2, scoring='accuracy')
    scores2 = cross_val_score(clf, X, y, cv=2, scoring='precision')
    scores3 = cross_val_score(clf, X, y, cv=2, scoring='recall')
    scores4 = cross_val_score(clf, X, y, cv=2, scoring='roc_auc')
    
    # The mean score and standard deviation of the score estimate
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
    print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
    print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
    print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))
    
    return 

def get_eval2(clf, X_train, y_train,X_test, y_test):
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(clf, X_test, y_test, cv=2, scoring='accuracy')
    scores2 = cross_val_score(clf, X_test, y_test, cv=2, scoring='precision')
    scores3 = cross_val_score(clf, X_test, y_test, cv=2, scoring='recall')
    scores4 = cross_val_score(clf, X_test, y_test, cv=2, scoring='roc_auc')
    
    # The mean score and standard deviation of the score estimate
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
    print("Cross Validation Precision: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
    print("Cross Validation Recall: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
    print("Cross Validation roc_auc: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std()))
    
    return  
  
# Function to get roc curve
def get_roc (y_test,y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="upper left")
    plt.show()
    return

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# fit, train and cross validate Decision Tree with training and test data 
def xgbclf(params, X_train, y_train,X_test, y_test):
  
    eval_set=[(X_train, y_train), (X_test, y_test)]
    
    model = XGBClassifier(**params).\
      fit(X_train, y_train, eval_set=eval_set, \
                  eval_metric='auc', early_stopping_rounds = 100, verbose=100)

    model.set_params(**{'n_estimators': model.best_ntree_limit})
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit) 
  
    abclf_cm = confusion_matrix(y_test,y_pred)
    print(abclf_cm)
    print(abclf_cm[0][0]/ (abclf_cm[0][0] + abclf_cm[1][0]))
    print (classification_report(y_test,y_pred) )
    print ('\n')
    print ("Model Final Generalization Accuracy: %.6f" %accuracy_score(y_test,y_pred) )
    
    y_pred_proba = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)[:,1] 
    get_roc (y_test,y_pred_proba)
    return model

def plot_featureImportance(model, keys):
  importances = model.feature_importances_

  importance_frame = pd.DataFrame({'Importance': list(importances), 'Feature': list(keys)})
  importance_frame.sort_values(by = 'Importance', inplace = True)
  importance_frame.tail(10).plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')

file = 'german.data'
names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

data = pd.read_csv(file,names = names, delimiter=' ')
print(data.shape)
print (data.columns)
data.head(10)

# Binarize the y output for easier use of e.g. ROC curves -> 0 = 'bad' credit; 1 = 'good' credit
data.classification.replace([1,2], [1,0], inplace=True)
# Print number of 'good' credits (should be 700) and 'bad credits (should be 300)
data.classification.value_counts()

#numerical variables labels
numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
           'existingcredits', 'peopleliable', 'classification']

# Standardization
numdata_std = pd.DataFrame(StandardScaler().fit_transform(data[numvars].drop(['classification'], axis=1)))

#categorical variables labels
catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
           'telephone', 'foreignworker']

d = defaultdict(LabelEncoder)

# Encoding the variable
lecatdata = data[catvars].apply(lambda x: d[x.name].fit_transform(x))

# print transformations
for x in range(len(catvars)):
    print(catvars[x],": ", data[catvars[x]].unique())
    print(catvars[x],": ", lecatdata[catvars[x]].unique())

#One hot encoding, create dummy variables for every category of every categorical variable
dummyvars = pd.get_dummies(data[catvars])
data_clean = pd.concat([data[numvars], dummyvars], axis = 1)
print(data_clean.shape)

X_clean = data_clean.drop('classification', axis=1)
y_clean = data_clean['classification']
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean,y_clean,test_size=0.2, random_state=1)

params2 = {'n_estimators':3000, 'objective': 'binary:logistic', 'learning_rate': 0.005, 'subsample':0.555, 'colsample_bytree':0.7,
           'min_child_weight':3, 'max_depth':8, 'n_jobs' : -1}
xgbclf(params2, X_train_clean, y_train_clean, X_test_clean, y_test_clean)



#Entire DataSet
'''
[[ 18  41]
 [  9 132]]
Acc = 0.75

flipping gender
[[ 19  40]
 [  9 132]]
Acc = 0.755
'''

#Anti - Classification
'''
Age = [[ 22,  37], 
       [ 14 ,127]];  0.745
       
['statussex_A91', 'statussex_A92', 'statussex_A93', 'statussex_A94']
Sex = [[ 19,  40]
       [  9, 132]];  0.755
'''

#Independence and Seperation
'''
Age >= 35  = [[ 2 23]
              [ 0 66]];  0.747
Age < 35   = [[16 29]
              [ 9 56]];  0.655

Sex_Female  =  [[ 19  40]
                [  9 132]];  0.755
Sex_Male  = [[ 22  37]
             [ 13 128]]; 0.75
'''

#Task 3
'''
Age >= 40  =  [[ 4  9]
               [ 4 43]];  0.783
Age < 40  =  [[12 34]
               [ 4 91]];  0.73
TPR = 0.5/0.75

Age >= 30  =  [[ 6 33]
               [ 3 84]];  0.714
Age < 30  = [[ 7 18]
             [10 40]];  0.627
TPR = 0.67/ 0.41


Age >= 38  =  [[ 2 14]
               [ 0 53]]; 0.797
Age < 38  = [[19 27]
             [ 3 83]];  0.772
TPR = 1/0.86

Age >= 25   =  [[ 14  26]
                [ 10 121]]; 0.789
Age < 25  =  [[ 3 11]
               [ 3 13]];  0.533
TPR = 0.58/0.5

Age >= 28  =   [[14 25]
                [ 5 98]];  0.789
Age < 28  =   [[ 8 11]
               [ 3 37]];  0.7627
TPR = 0.73/0.73
'''

