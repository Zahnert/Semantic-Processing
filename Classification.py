import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import var
from math import sqrt
import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats

#load dataframe for classification
df = pd.read_csv(f"{SUB_DIR}/XY/degree.csv")


# shuffle
dataset = df
dataset = shuffle(dataset)


# random undersampling to obtain equal sample sizes in both groups
count_class_0, count_class_1 = dataset["Cluster"].value_counts()
# Divide by class
df_class_0 = dataset[dataset['Cluster'] == 1.0]
df_class_1 = dataset[dataset['Cluster'] == 0.0]
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under["Cluster"].value_counts())

df_test_under["Cluster"].value_counts().plot(kind='bar', title='Count (target)');
dataset = df_test_under
dataset = shuffle(dataset)

# Perform Train-test-split
train, test = train_test_split(dataset, test_size=0.2, random_state = 1)

# save these, as they were generated from a random sample. 
#Testing of saved classifiers on different random sets would lead to falsely high ROC, as previously seen data would be used for testing.
train.to_csv(f"{SUB_DIR}/XY/train.csv")
test.to_csv(f"{SUB_DIR}/XY/test.csv")


### FEATURE SELECTION VIA PERMUTATION TESTING ###

def cohend(d1, d2):
    # calculate the size of samples, code adopted from machinelearningmastery.com
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

#Note that features are selected via analysis of the train set, while the test-set remains left aside.
typical = train.loc[train['Cluster'] == 1] # Cluster = our label of 1 = typical lateralization
atypical = train.loc[train['Cluster'] == 0]
typical = typical.drop(['LI', 'Cluster'], axis=1)
atypical = atypical.drop(['LI', 'Cluster'], axis=1)

# Same as in Statistics.py
k = np.zeros(2,)
for i in range(0,96):
    result = stats.ttest_ind(typical.iloc[:,i], atypical.iloc[:,i], equal_var = False, permutations=100000)
    res = np.array(result)
    k = np.vstack((k, res))
k = np.delete(k, 0, 0)

cohend_vec = np.zeros(96,)
for i in range(0,96):
    d1 = typical.iloc[:,i].values
    d2 = atypical.iloc[:,i].values
    cohen = cohend(d1, d2)
    cohend_vec[i] += cohen
   
feature_cols = typical.columns
np.set_printoptions(suppress=True)
ttest_dict = dict(zip(feature_cols, zip(k[:, 0], k[:, 1], cohend_vec)))
ttest_df = pd.DataFrame.from_dict(ttest_dict, orient='index')
ttest_df.rename(columns = {0:'t_stat', 1:'p_val', 2:'cohen_d'}, inplace = True)
ttest_df.sort_values(by=['p_val'], ascending=True).head(10)


# Here, we select significant features (uncorrected) according to the results from above
# Example for node degree  
train = train[['ctx-rh-post-middletemporal', 'ctx-rh-posteriorcingulate', 'ctx-rh-post-fusiform', 
               'ctx-lh-postcentral', 'ctx-lh-ant-transversetemporal', 'ctx-rh-frontalpole',
               'ctx-lh-parsorbitalis', 'ctx-rh-ant-transversetemporal', 'LI', 'Cluster']]
test = test[['ctx-rh-post-middletemporal', 'ctx-rh-posteriorcingulate', 'ctx-rh-post-fusiform', 
               'ctx-lh-postcentral', 'ctx-lh-ant-transversetemporal', 'ctx-rh-frontalpole',
               'ctx-lh-parsorbitalis', 'ctx-rh-ant-transversetemporal', 'LI', 'Cluster']]


### BACK TO CLASSIFIERS ###

# Generate X and y
y_train = train["Cluster"].values
X       = train.drop(["Cluster", "LI"], axis=1)
X_train = X.to_numpy()
y_test = test["Cluster"].values
X_test = test.drop(["Cluster", "LI"], axis=1)
X_test = X_test.to_numpy()

#Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Split off validation set
X_val = X_train[0:50,:]
y_val = y_train[0:50]
X_train = X_train[50:,:]
y_train = y_train[50:,]

# set feature shape
feature_vector_length = 8
input_shape = (feature_vector_length,)
print(f'Feature shape: {input_shape}')

# function for model generation, neural network
def make_model(input_shape, neurons_1, neurons_2, lr):

    model = Sequential()
    model.add(Dense(neurons_1, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(neurons_2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.AUC(from_logits=True),
                           'accuracy'])
    return model

# Call make model and fit && validate
model = make_model(input_shape=input_shape,neurons_1 = 17, neurons_2 = 16, lr = 0.001)
model.fit(
    X_train,
    y_train,
    validation_data = (X_val, y_val),
    epochs=50,
    verbose=1)
 

# after validation: test model
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - AUC-ROC: {test_results[1]}% -  bin accuracy: {test_results[2]}%')



### TRAIN A RANDOM FOREST AND PLOT AUC OF BOTH RF AND NN ###

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from numpy import mean

# for validation of the RF, we use cross-validation, as is usual. 
# Note that this leads to different training sets
X_train2 = np.vstack((X_train, X_val))
y_train2 = np.append(y_train, y_val)

#Our Random Forest Classifier; set hyperparameters to those of the best estimator from GridSearch
rf = RandomForestClassifier(criterion='entropy',
                             max_depth=3, max_features=1, min_samples_split=2)

def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    return scores


# Hyperparameter tuning via Grid search
param_grid = {'criterion':['gini', 'entropy'], 
              'max_features':['sqrt', 1, 2, 3], 
              'max_depth':[2, 3], 
              'min_samples_split':[2,3,4,5,6], 
              'n_estimators':[100,200,300]} #forest

cross_val = KFold(n_splits=5, random_state=42)
rf_grid_search = GridSearchCV(rf, param_grid, cv=cross_val, scoring='roc_auc', n_jobs=-1)
rf_grid_search.fit(X_train2, y_train2)
rf_grid_search.best_estimator_

# Validation
scores = evaluate_model(X_train2, y_train2, clf)
# summarize performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), np.std(scores)))

rf.fit(X_train2, y_train2)


### PLOT AUC OF BOTH CLASSIFIERS ###

# load the neural network
x = keras.models.load_model(f"{SUB_DIR}/XY/degree_model")

from sklearn.metrics import roc_curve

y_pred_keras = x.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test, y_pred_proba)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
auc_forest = auc(fpr_forest, tpr_forest)

### PLOT AUC ###

from matplotlib import pyplot as plt              
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='NN (AUC = {:.3f})'.format(auc_keras))
plt.plot(fpr_forest, tpr_forest, label='RF (AUC = {:.3f})'.format(auc_forest))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(f"{SUB_DIR}/XY/roc_degree.tif", dpi=600)
plt.show()

# Get Test Accuracy of RF
test_score = rf.score(X_test, y_test)
test_score

# Of course, models and stats were saved, which is omitted here
# Confidence Intervals for AUC were calculated using the DeLong method, adopted from https://github.com/yandexdataschool/roc_comparison

#########################################################

### CI for Accuracy were calculated using bootstrapping ####
# adopted and modified from https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/


from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import median
from numpy import percentile
from numpy.random import seed
from numpy import randint

seed(42)

y_pred_rf = rf.predict(X_test)

score = accuracy_score(y_test, y_pred_rf)
scores = list()

for i in range(1000):
    # prepare train and test sets
    
    indices = randint(0,len(y_test),len(y_test))
    score = accuracy_score(y_test[indices], y_pred_rf[indices])
    scores.append(score)
    
print('50th percentile (median) = %.4f' % median(scores))
# calculate 95% confidence intervals (100 - alpha)
alpha = 5.0
# calculate lower percentile (e.g. 2.5)
lower_p = alpha / 2.0
# retrieve observation at lower percentile
lower = max(0.0, percentile(scores, lower_p))
print('%.1fth percentile = %.4f' % (lower_p, lower))
# calculate upper percentile (e.g. 97.5)
upper_p = (100 - alpha) + (alpha / 2.0)
# retrieve observation at upper percentile
upper = min(1.0, percentile(scores, upper_p))
print('%.1fth percentile = %.4f' % (upper_p, upper))
