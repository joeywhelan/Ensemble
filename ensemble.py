'''
Created on Sep 8, 2017

@author: Joey Whelan
'''

import logging.config
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
from time import time
from sample import Sample
from scipy.stats import pearsonr
from loader import load
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
from collections import OrderedDict

MINORITY_CLASS = 0 #represents loan defaults
MINORITY_POS = 0 #position within prediction probability array of the loan default probability
LABEL_COL = 'loan_status' #header of the label column within a data frame containing loan features

def vfunc(probability, prediction, threshold):
    """Simple function for thresholding predictions.  
        
        Args:
            probability: The probability of the minority class
            prediction:  The class prediction of the classifier
            threshold:   Probability threshold above which the minority class is predicted
        
        Returns:
            Class prediction, adjusted for the threshold
        
        Raises:
            None
    """
    if probability >= threshold:
        return MINORITY_CLASS
    else:
        return prediction
    
def lvl1_fit(clf, name, features_train, labels_train):
    """Function to be called within a multiprocessing pool to generate the 1st level predictions in a stacking
        ensemble.
        
        Args:
            clf: SKLearn Classifier
            name:  String representing name of classifier
            features_train:   Array of training features
            labels_train:  Array of training labels
        
        Returns:
            Dict of fitted classifer and its name
        
        Raises:
            None
    """
    logging.debug('Entering lvl1_fit() {}'.format(name))
    ti = time()
    fittedclf = clf.fit(features_train, labels_train)
    logging.debug('{} fit time: {:0.4f}'.format(name, time()-ti))
    joblib.dump(fittedclf, './models/' + name + '.pkl') #cache the fitted model to disk
    logging.debug('Exiting lvl1_fit() {}'.format(name))
    return {'name': name, 'fittedclf': fittedclf}

def lvl2_fit(clf, name, fold, test_idx, col_loc, features_train, labels_train, features_test):
    """Function to be called within a multiprocessing pool as step in a k-fold cross-validation loop 
        to generate the predictions from trained, 1st level classifiers.  
        Those predictions are then used as training features for a logistic classifier.
        
        Args:
            clf: SKLearn Classifier
            name:  String representing name of classifier
            fold: Integer representing fold number.  Used for logging/debug.
            test_idx:  Array of indices representing the frame rows where the predictions should be inserted.
            col_loc: column index where predictions should be inserted
            features_train:   Array of training features
            labels_train:  Array of training labels
            features_test: Array of test features.  Predictions are made on these to feed the 2nd Level Logistic
            classifer as training data.
        
        Returns:
            Dict of classifier name, row indices, column index, and predictions
        
        Raises:
            None
    """
    logging.debug('Entering lvl2_fit() {} fold {}'.format(name, fold))
    ti = time()
    clf.fit(features_train, labels_train)
    logging.debug('{} fold {} fit time: {:0.4f}'.format(name, fold, time()-ti))
    preds = clf.predict_proba(features_test)[:, MINORITY_POS]
    logging.debug('Exiting lvl2_fit() {} fold {}'.format(name, fold))
    return {'name': name, 'test_idx' : test_idx, 'col_loc' : col_loc, 'preds' : preds}
               
class Ensemble(object):
    """
     Class that implements voting and stacking techniques for ensemble classification.
    """    
    def __init__(self, algorithm='stack', threshold=.5):
        """Class initializer.  Sets up 1st level classifiers.
        
        Args:
            algorithm: Ensemble type.  Either 'vote' or 'stack'
            threshold:  Value for custom thresholding of the minority class
        
        Returns:
            None
        
        Raises:
            None
        """
        logging.config.fileConfig('./logging.conf')
        
        if algorithm == 'stack' or algorithm == 'vote':
            self.algorithm = algorithm
        else:
            raise Exception('invalid algorithm type')
        
        self.threshold = threshold
        rbm = Pipeline(steps=[('minmax', MinMaxScaler()), \
                                   ('rbm', BernoulliRBM(learning_rate=0.001,n_iter=20,n_components=100)), \
                                   ('logistic', LogisticRegression(C=6.0))])
        svd = Pipeline(steps=[('svd', TruncatedSVD(n_components=20)), 
                              ('logistic', LogisticRegression(C=6.0))])
        gbc = GradientBoostingClassifier(learning_rate=.1, max_depth=5, n_estimators=36)
        mlp = Pipeline(steps=[('stdScaler', StandardScaler()), \
                                   ('mlp', MLPClassifier(alpha=10.0**-7, random_state=1, early_stopping=True, \
                                        hidden_layer_sizes=(20,10,10), max_iter=1000, batch_size=128))]) 
        
        #Object variable holding the classifiers.  Note this has been defined as an OrderedDict.  Maintaing
        #order of the classifiers is mandatory.
        self.estimators=OrderedDict([('gbc', gbc), ('mlp', mlp), ('rbm', rbm), ('svd', svd)])
            
    def classification_report(self, name, labels_test, preds):
        """Public helper function for printing classification score
        
        Args:
            name: Classifier name
            threshold:  Test labels
            preds: Predictions based on test features
        
        Returns:
            None
        
        Raises:
            None
        """
        print('{} Classification Report'.format(name))
        print(classification_report(labels_test, preds, target_names=['Default', 'Paid'])) 
    
    def confusion_matrix(self, name, labels_test, preds): 
        """Public helper function for printing classification confusion matrix
        
        Args:
            name: Classifier name
            threshold:  Test labels
            preds: Predictions based on test features
        
        Returns:
            None
        
        Raises:
            None
        """
        print('{} Confusion Matrix ({} samples): '.format(name, len(labels_test)))
        print(confusion_matrix(labels_test, preds))
        
    def fit(self, features_train, labels_train):
        """Public interface to fit the ensemble
        
        Args:
            features_train: Array of training features
            lablels_train:  Array of training labels
           
        Returns:
            None
        
        Raises:
            None
        """
        logging.debug('Entering fit()')
        if self.algorithm == 'vote':
            self.__fit_vote(features_train, labels_train)
        else:
            if self.algorithm == 'stack':
                self.__fit_stack(features_train, labels_train)
        logging.debug('Exiting fit()')
    
    def predict(self, features):
        """Public interface to generate predictions from the ensemble
        
        Args:
            features: Array of features
           
        Returns:
            Array of predctions
        
        Raises:
            None
        """
        logging.debug('Entering predict()')
        preds = None
        if self.algorithm == 'vote':
            preds = self.__predict_vote(features)
        else:
            if self.algorithm == 'stack':
                preds = self.__predict_stack(features)
        logging.debug('Exiting predict()')
        return preds

    def test(self, features_train, labels_train, features_test, labels_test):
        """Public helper function to display test results of 1st level predictors and ensemble
        
        Args:
            features_train: Array of training features
            labels_train: Array of training labels
            features_test: Arrays of test features
            labels_test: Arrays of test labels
           
        Returns:
            None
        
        Raises:
            None
        """
        pool = mp.Pool(processes=mp.cpu_count())
        results = []

        for name, clf in self.estimators.items():
            try:
                self.estimators[name] = joblib.load('./models/' + name + '.pkl')
            except FileNotFoundError:  
                logging.debug('{} not pickled'.format(name))    
                results.append(pool.apply_async(lvl1_fit, args=(clf, name, features_train, labels_train)))           
           
        pool.close()
        pool.join() 
        for result in results:
            item = result.get()
            name = item['name']
            self.estimators[name] = item['fittedclf']
        
        #Print confusion matrix and score for each clf.  
        corr_list = []
        clf_list = []
        for name, clf in self.estimators.items():
            preds = clf.predict(features_test)
            self.confusion_matrix(name, labels_test, preds)
            print()
            self.classification_report(name, labels_test, preds)
            corr_list.append((name, preds))
            clf_list.append(name)
        
        #Print a matrix of correlations between clfs
        frame = pd.DataFrame(index=clf_list, columns=clf_list)
    
        for pair in itertools.combinations(corr_list,2):
            res = pearsonr(pair[0][1],pair[1][1])[0]
            frame[pair[0][0]][pair[1][0]] = res
            frame[pair[1][0]][pair[0][0]] = res
        frame['mean'] = frame.mean(skipna=True,axis=1)
        pd.options.display.width = 180
        print('Correlation Matrix')
        print(frame) 
    
    #Private class variable containing vectorized, threshold prediction function       
    __custom_predict = np.vectorize(vfunc, otypes=[np.int])
    
    def __fit_stack(self, features_train, labels_train):
        """Private function implementing the classifier fit for a stacking ensemble
        
        Args:
            features_train: Array of training features
            labels_train: Array of training labels
            
        Returns:
            None
        
        Raises:
            None
        """
        logging.debug('Entering __fit_stack()')
        
        pool = mp.Pool(processes=mp.cpu_count())
        results = [] #array for holding the result objects from the pool processes
        
        #fit 1st level estimators with a multiprocessing pool of workers
        for name, clf in self.estimators.items():
            try:
                self.estimators[name] = joblib.load('./models/' + name + '.pkl')
            except FileNotFoundError:  
                logging.debug('Level 1: {} not pickled'.format(name))    
                results.append(pool.apply_async(lvl1_fit, args=(clf, name, features_train, labels_train)))           
           
        pool.close()
        pool.join() 
       
        for result in results:
            item = result.get()
            name = item['name']
            self.estimators[name] = item['fittedclf'] #reassign a fitted clf to the estimator dictionary
        
        #fit 2nd level estimator with a multiprocessing pool of workers that perform a k-fold cross-val of 
        #training data
        pool = mp.Pool(processes=mp.cpu_count())
        del results[:]
        try:
            self.lrc = joblib.load('./models/lrc.pkl') #try to load the 2nd level estimator from disk
        except FileNotFoundError: #2nd level estimator not fitted yet
            logging.debug('Level 2: LRC not pickled') 
            folds = list(StratifiedKFold(n_splits=5).split(features_train, labels_train)) 
            #define a frame for holding the k-fold test results of the 1st level classifiers
            lvl2_frame = pd.DataFrame(index=range(0,len(features_train)), columns=list(self.estimators.keys()))  
            lvl2_frame[LABEL_COL] = labels_train  
             
            #launch multiprocessing pool workers (1 per fold) that fit 1st level classifers and perform
            #predictions that become the training data for the 2nd level classifier (Logistic Regression)   
            for name,clf in self.estimators.items():
                fold = 1
                for train_idx, test_idx in folds:
                    X_train, X_test = features_train[train_idx], features_train[test_idx]
                    Y_train = labels_train[train_idx]
                    col_loc = lvl2_frame.columns.get_loc(name)
                    results.append(pool.apply_async(lvl2_fit, args=(clf, name, fold, test_idx, \
                                                                    col_loc, X_train, Y_train, X_test)))
                    fold = fold + 1
            pool.close()
            pool.join() 
           
            #fetch worker results and put them into a frame that will be used to train a 2nd Level/Logistic
            #regression classifier
            for result in results:
                item = result.get()
                name = item['name']
                test_idx = item['test_idx']
                col_loc = item['col_loc']
                preds = item['preds']
                lvl2_frame.iloc[test_idx, col_loc] = preds
                
            #lvl2_frame.to_csv('./models/lvl2frame.csv')
            self.lrc = LogisticRegression(C=2.0)
            ti = time()
            X = lvl2_frame.drop(LABEL_COL, axis=1).values
            Y = lvl2_frame[LABEL_COL].values
            self.lrc.fit(X, Y)     
            logging.debug('LRC fit time: {:0.4f}'.format(time()-ti))
            joblib.dump(self.lrc, './models/lrc.pkl')  #cache the Logistical Regressor to disk
        logging.debug('Exiting __fit_stack()') 
       
    def __fit_vote(self, features_train, labels_train):
        """Private function implementing the classifier fit for a voting ensemble.  Wrapper around the
        SKLearn voting classifier.
        
        Args:
            features_train: Array of training features
            labels_train: Array of training labels
            
        Returns:
            None
        
        Raises:
            None
        """
        logging.debug('Entering __fit_vote()')
        try:
            self.voteclf = joblib.load('./models/voteclf.pkl')
        except FileNotFoundError: 
            ti = time() 
            self.voteclf = VotingClassifier(estimators=list(self.estimators.items()), voting='soft',n_jobs=-1)      
            self.voteclf.fit(features_train, labels_train)
            logging.debug('fit time: {:0.4f}'.format(time()-ti))
            joblib.dump(self.voteclf, './models/voteclf.pkl') #cache the fitted model to disk
        logging.debug('Exiting __fit_vote()')
    
    def __predict_stack(self, features):
        """Private function that collects the 1st level classifier probabilities and then uses them as
        the feature set to a 2nd level classifier (Logistic Regression).
        
        Args:
            features: Array of features
            
        Returns:
            Array of predictions
        
        Raises:
            None
        """
        logging.debug('Entering __predict_stack()')
        lvl1_frame = pd.DataFrame()
        #1st level predictions
        for name, clf in self.estimators.items():
            lvl1_frame[name] = clf.predict_proba(features)[:, MINORITY_POS]
            
        #2nd level predictions
        preds = self.__predict_with_threshold(self.lrc, lvl1_frame.values)
        
        logging.debug('Exiting __predict_stack()')
        return preds
        
    def __predict_vote(self, features):
        """Private function that is a wrapper for the SKLearn voting classifier prediction method.
        
        Args:
            features: Array of features
            
        Returns:
            Array of predictions
        
        Raises:
            None
        """
        logging.debug('Entering __predict_vote()')
        preds = self.__predict_with_threshold(self.voteclf, features)
        logging.debug('Exiting __predict_vote()')
        return preds
    
    def __predict_with_threshold(self, clf, features):
        """Private function that wraps a classifier's predict method with functionality to implement
        thresholding for the minority class
        
        Args:
            clf: SKLearn classifier
            features: Array of features
            
        Returns:
            Array of predictions
        
        Raises:
            None
        """
        logging.debug('Entering __predict_with_threshold()')
        ti = time()
        predictions = Ensemble.__custom_predict(clf.predict_proba(features)[:, MINORITY_POS], \
                                                clf.predict(features), self.threshold)
        logging.debug('prediction time: {:0.4f}'.format(time()-ti))
        logging.debug('Exiting __predict_with_threshold()')
        return predictions
                  
if __name__ == '__main__':
    '''Sample/test calls of the various public functions
    '''
    logging.config.fileConfig("logging.conf")
    frame = load() #Pandas dataframe (cleaned) of Lending Club historical data
    test_size = .1
    
    #Pull a test set from the frame
    tempframe = frame.sample(n=int(len(frame.index)*test_size), replace=False, random_state=2016)
    labels_test = tempframe[LABEL_COL].values
    features_test = tempframe.drop(LABEL_COL, axis=1).values 
    frame.drop(tempframe.index.tolist(), inplace=True)

    #Balance the minority/majority classes with simple up-sampling
    bal = Sample(label_col='loan_status', min_class=MINORITY_CLASS, direction='up', mult='balanced')   
    balframe = bal.balance(frame)
    labels_train = balframe[LABEL_COL].values
    features_train = balframe.drop(LABEL_COL, axis=1).values 
    
    #Show test results of the various classifiers in isolation
    ens = Ensemble()
    ens.test(features_train, labels_train, features_test, labels_test)
    
    #Show test results of the classifiers in a voting ensemble
    ens = Ensemble(algorithm='vote')
    ens.fit(features_train, labels_train)
    preds = ens.predict(features_test)
    ens.confusion_matrix('vote', labels_test, preds)
    ens.classification_report('vote', labels_test, preds)
    
    #Show test results of the classifiers in a stacking ensemble
    ens = Ensemble(algorithm='stack', threshold=.3)
    ens.fit(features_train, labels_train)
    preds = ens.predict(features_test)
    ens.confusion_matrix('stack', labels_test, preds)
    ens.classification_report('stack', labels_test, preds)
    
   
    
