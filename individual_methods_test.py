from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
import scipy.optimize
import numpy as np
from fancyimpute import SoftImpute
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

#necessary to load R package MCPanel 
#to succesfully load the package, change the directory of R_HOME to where R is installed 
#and similarly set R_USER to where the python package rpy2 is installed.
import os
#os.environ['R_HOME'] = r"C:\Program Files\R\R-3.4.3"
#os.environ['R_USER'] = r"C:\Users\zhaon\AppData\Local\Programs\Python\Python36\Lib\site-packages\rpy2"
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('library(MCPanel)')

def MC(outcome_matrix):
    
    #uses singular value thersholding ala Mazumder
    X_incomplete = outcome_matrix
    X_incomplete[-1,-1] = np.nan
    completed_matrix = SoftImpute().fit_transform(X_incomplete)
    
    predicted_outcome = completed_matrix[-1,-1]
    
    return predicted_outcome

def MC_fixed_effect(outcome_matrix,axis=1):
    
    #uses singular value thersholding ala Mazumder
    if axis == 1:
        fixed_effect = np.mean(outcome_matrix,1)
        X_incomplete = (outcome_matrix.T-fixed_effect).T
        X_incomplete[-1,-1] = np.nan
        completed_matrix = (SoftImpute().fit_transform(X_incomplete).T+fixed_effect).T
    
    predicted_outcome = completed_matrix[-1,-1]
    
    return predicted_outcome

def MC_fe(outcome_matrix):
    
    #uses MCPanel R package a la Athey et al.
       
    units,periods = outcome_matrix.shape
    Br = robjects.r.matrix(outcome_matrix, nrow=units, ncol=periods)
    robjects.r.assign('true_mat', Br)

    predicted_outcome=robjects.r('''
       
            #library(MCPanel)

            # create mask
            units = dim(true_mat)[1]
            periods = dim(true_mat)[2]

            mask <- array(1,dim=c(units,periods))
            mask[units,periods] <- 0

            obs_mat <- true_mat * mask

            model_with_both <- mcnnm_cv(obs_mat, mask, to_estimate_u = 1, to_estimate_v = 1)

            model_with_both$est <- model_with_both$L + replicate(periods,model_with_both$u) + t(replicate(units,model_with_both$v))

            model_with_both$est[units,periods]
            ''')
    
    return float(predicted_outcome)
    

def HR(outcome_matrix,a=0.015,b=1,cv=False):
    
    #outcome matrix has dimension units*periods
    X = outcome_matrix[:-1,:-1]
    y = outcome_matrix[:-1,-1]
    
    #need to reshape features into 2D array 
    X_predict = outcome_matrix[-1,:-1].reshape(1,-1)
    
    if cv == True:
    #elastic net with cross validation
        regr = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],cv=min(10,X.shape[0]-1))
        regr.fit(X,y)
        a = regr.alpha_
        b = regr.l1_ratio_
        predicted_outcome = regr.predict(X_predict)[0]
        return predicted_outcome, a, b
    else:
    #elastic net with predetermined hypterparameters
        regr = ElasticNet(alpha=a,l1_ratio=b)
        regr.fit(X,y)
        predicted_outcome = regr.predict(X_predict)[0]
        return predicted_outcome, regr.coef_, regr.intercept_
    

def VT(outcome_matrix,a=0.001,b=1,cv=False):
    
    #outcome matrix has dimension units*periods
    X = outcome_matrix[:-1,:-1].T
    y = outcome_matrix[-1,:-1].T
    
    #need to reshape features into 2D array 
    X_predict = outcome_matrix[:-1,-1].reshape(1,-1)
    
    if cv == True:
    #elastic net with cross validation
        regr = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],cv=min(10,X.shape[0]-1))
        regr.fit(X,y)
        a = regr.alpha_
        b = regr.l1_ratio_
        predicted_outcome = regr.predict(X_predict)[0]
        return predicted_outcome, a, b
    else:
    #elastic net with predetermined hypterparameters
        regr = ElasticNet(alpha=a,l1_ratio=b)
        regr.fit(X,y)
        predicted_outcome = regr.predict(X_predict)[0]
        return predicted_outcome, regr.coef_, regr.intercept_


def Abadie(outcome_matrix):
    
    #outcome matrix has dimension units*periods
    X = outcome_matrix[:-1,:-1]
    y = outcome_matrix[-1,:-1].T
    X_predict = outcome_matrix[:-1,-1].reshape(1,-1)

    #use non-negative regression and impose summation constraint
    #by adding a sample with constant large feature, then adding an extra entry
    #of the same large number to the outcome vector 
    
    X_train = X.T
    samples, features = X_train.shape
    magnitude = 1e5
    constant_feature = np.ones((1,features))*magnitude
    X_train = np.concatenate((X_train, constant_feature), axis=0)
    y_train = np.zeros((samples+1,))
    y_train[-1] = magnitude
    y_train[:-1] = y
            
    coeff, norm = scipy.optimize.nnls(X_train, y_train)

    predicted_outcome = np.dot(X_predict,coeff)[0]
    
    return predicted_outcome

def DID(outcome_matrix):
    #estimate 2-way fixed effect model using linear regression with indicators
    
    units,periods = outcome_matrix.shape
    unit_id = np.identity(units)
    period_id = np.identity(periods)
    
    #construct the design matrix 
    #top matrix are indicators of which unit
    top = np.repeat(unit_id, periods, axis=1)
    #bottom matrix are indicators of which period
    bottom = np.tile(period_id, units)
    #stack the two indicator matrices
    X = np.concatenate((top, bottom), axis=0)
    
    #X_train has dimension (unit*period-1)*(unit+period)
    X_train = X[:,:-1].T
    #X_predict has dimension 1*(unit+period)
    X_predict  = X[:,-1].reshape(1,-1)
    #y has dimension unit*period-1
    y = outcome_matrix.reshape(1,-1)
    y = np.ravel(y[:,:-1])
    
    #elastic net with cross validation
    #regr = ElasticNetCV(cv=3)
    
    #linear regression
    regr = LinearRegression()
    regr.fit(X_train,y)
    
    predicted_outcome = regr.predict(X_predict)[0]
   
    return predicted_outcome

def restricted_regression(design_matrix,outcome):
    
    samples, features = design_matrix.shape
    magnitude = 1e5
    constant_feature = np.ones((1,features))*magnitude
    X_train = np.concatenate((design_matrix, constant_feature), axis=0)
    y_train = np.zeros((samples+1,))
    y_train[-1] = magnitude
    y_train[:-1] = outcome
            
    coeff, norm = scipy.optimize.nnls(X_train, y_train)
    
    return coeff

#scikit learn neural network with cross validation
def neural_net_cv(X,y,cv_layers=[(50,),(20,)],cv_alpha=[0.001,0.01],cv=5):

    cross_validation = StratifiedShuffleSplit(n_splits=cv, test_size=1/cv)
    param_grid = dict(hidden_layer_sizes=cv_layers, alpha=cv_alpha)
    neural_regressor = MLPRegressor()
    grid = GridSearchCV(neural_regressor, param_grid=param_grid, cv=cv, verbose=1, n_jobs=2)
    grid.fit(X, y)
    
    return grid.best_params_.get('hidden_layer_sizes'),grid.best_params_.get('alpha')
        
    
    
    