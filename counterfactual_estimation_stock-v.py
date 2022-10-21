import numpy as np
import individual_methods as estimate
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
import pandas as pd
import time

#read in data
data = pd.read_csv('geithner_outcome.csv')

units = 70
periods = 70
GDP = data.values[:units,-periods:].astype(float)*1e2

#specify the number of test and train periods
test_periods = 10
train_periods = periods-test_periods

#record RMSE of predictions by 8 methods
RMSE_test = np.zeros((units,test_periods,8))

#ensemble weights

#HR+VT restricted
ensem1_weights = np.zeros((units,2))

#HR+VT+MC restricted
ensem2_weights = np.zeros((units,3))

#HR+VT+Abadie+MC restricted
ensem3_weights = np.zeros((units,4))

#HR+VT+Abadie restricted
ensem4_weights = np.zeros((units,3))

#to calculate correlation of predictions by different methods
Epsilon = np.zeros((units,test_periods,4))

#record the regularization strengths obtained from cross validation
HR = np.zeros((units,2))
VT = np.zeros((units,2))

#in each test period obtain predictions of individual methods
#then obtain weights and predictions of ensemble methods

Y = GDP[:,:-test_periods]

for current_unit in range(units):
    
    #shift current unit to the last row
    Y0 = np.roll(Y,-current_unit-1,0)
    
    #to save computation time, obtain regularization from cross-validation
    #for last period of current unit and use that for all train periods
    _, HR_alpha, HR_l1_ratio = estimate.HR(Y0,cv=True)
    _, VT_alpha, VT_l1_ratio = estimate.VT(Y0,cv=True)
    
    #record the regularization strengths obtained from cross validation
    HR[current_unit,0] = HR_alpha
    HR[current_unit,1] = HR_l1_ratio
    VT[current_unit,0] = VT_alpha
    VT[current_unit,1] = VT_l1_ratio
    
for test_period in range(test_periods): 

    Y0 = GDP[:,:periods-test_periods+test_period+1]
    
    #record predictions of individual methods and actual outcome for 
    #regression ensemble weighting
    individual_predictions = np.zeros((units,4))
    actual_outcomes = np.zeros(units,)
    
    for current_unit in range(units):
        
        HR_alpha,HR_l1_ratio = HR[current_unit,:]
        VT_alpha,VT_l1_ratio = VT[current_unit,:]
        
        #shift current unit to the last row
        Y_shift = np.roll(Y0,-current_unit-1,0)
        
        actual_outcomes[current_unit] = Y_shift[-1,-1]
        
        #obtain RMSE for ensemble weighting
        
        #METHOD-1 MC
        individual_predictions[current_unit,0] = estimate.MC(Y_shift)
        
        #METHOD-2 HR
        individual_predictions[current_unit,1] = estimate.HR(Y_shift,a=HR_alpha,b=HR_l1_ratio)
        #individual_predictions[current_unit,1] = estimate.HR(Y_shift,cv=True)
        
        #METHOD-3 VT
        individual_predictions[current_unit,2] = estimate.VT(Y_shift,a=VT_alpha,b=VT_l1_ratio)
        #individual_predictions[current_unit,2] = estimate.VT(Y_shift,cv=True)
        
        #METHOD-4 Abadie
        individual_predictions[current_unit,3] = estimate.Abadie(Y_shift)
        
    Epsilon[:,test_period,:] = (individual_predictions.T-actual_outcomes).T
        
    RMSE_test[:,test_period,:4] = np.absolute(Epsilon[:,test_period,:])
        
    #obtain ensemble weights and predictions for regression ensemble methods
    
    for current_unit in range(units):
        
        mask = (np.arange(units) != current_unit)
        
        cv_predictions = individual_predictions[mask,:]
        cv_outcomes = actual_outcomes[mask]
        
        actual_outcome = actual_outcomes[current_unit]
    
        #HR+VT restricted
        ensem1_weights[current_unit,:] = estimate.restricted_regression(cv_predictions[:,1:3],cv_outcomes)
        
        ensem_prediction = np.dot(ensem1_weights[current_unit,:],individual_predictions[current_unit,1:3])
        RMSE_test[current_unit,test_period,5] = np.absolute(ensem_prediction-actual_outcome)
        
        #HR+VT+MC restricted
        ensem2_weights[current_unit,:] = estimate.restricted_regression(cv_predictions[:,0:3],cv_outcomes)

        ensem_prediction = np.dot(ensem2_weights[current_unit,:],individual_predictions[current_unit,0:3])
        RMSE_test[current_unit,test_period,4] = np.absolute(ensem_prediction-actual_outcome)
        
        #HR+VT+Abadie+MC restricted
        ensem3_weights[current_unit,:] = estimate.restricted_regression(cv_predictions[:,0:],cv_outcomes)

        ensem_prediction = np.dot(ensem3_weights[current_unit,:],individual_predictions[current_unit,0:])
        RMSE_test[current_unit,test_period,6] = np.absolute(ensem_prediction-actual_outcome)
        
        #HR+VT+Abadie restricted
        ensem4_weights[current_unit,:] = estimate.restricted_regression(cv_predictions[:,1:],cv_outcomes)
 
        ensem_prediction = np.dot(ensem4_weights[current_unit,:],individual_predictions[current_unit,1:])
        RMSE_test[current_unit,test_period,7] = np.absolute(ensem_prediction-actual_outcome)

#calculate RMSE by averaging over test periods, i.e. axis 1        
RMSE_final = np.squeeze(np.sqrt(np.sum(np.square(RMSE_test),1)/test_periods))
RMSE_final = (RMSE_final.T/np.std(GDP,axis=1)).T
overall = np.sqrt(np.sum(np.square(RMSE_final),0)/units)

RMSE_output = np.concatenate((RMSE_final,np.reshape(overall,(1,-1))),axis=0)

#calculate correlation and covariance of individual methods
cov_input = np.reshape(Epsilon.transpose(),(4,-1))

covariance_matrix = np.cov(cov_input)
correlation_matrix = np.corrcoef(cov_input)

index1 = list(np.arange(units))
index1 = index1.append('mean')
columns1 = ['MC','HR','VT','Abadie','HR+VT','MC+HR+VT','MC+HR+VT+Abadie','HR+VT+Abadie']
RMSE_print = pd.DataFrame(np.around(RMSE_output,5),index=index1,columns=columns1)  

index2 = np.arange(units)
columns2 = ['MC','HR','VT','Abadie']

corr_print = pd.DataFrame(correlation_matrix,index=columns2,columns=columns2)
ensem1_print = pd.DataFrame(ensem1_weights,index=index2,columns=['HR','VT'])
ensem2_print = pd.DataFrame(ensem2_weights,index=index2,columns=['MC','HR','VT'])
ensem3_print = pd.DataFrame(ensem3_weights,index=index2,columns=['MC','HR','VT','Abadie'])
ensem4_print = pd.DataFrame(ensem4_weights,index=index2,columns=['HR','VT','Abadie'])
    
RMSE_print.to_csv('RMSE_70.csv')
corr_print.to_csv('corr_70.csv')
ensem1_print.to_csv('ensem1_70.csv')
ensem2_print.to_csv('ensem2_70.csv')
#ensem3_print.to_csv('ensem3_10.csv')
#ensem4_print.to_csv('ensem4_10.csv')