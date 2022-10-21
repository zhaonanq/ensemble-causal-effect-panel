import numpy as np
import individual_methods_weight as estimate
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
import pandas as pd
import time

#read in data
data = pd.read_csv('GDP_data.csv')

#select the last 100 periods and then scale data by 1e-8
GDP = data.values[:,-100:].astype(float)*1e-8
#GDP = np.log(data.values[:,-100:].astype(float))
units, periods = GDP.shape

#GDP growth rate
#GDP = data.values[:,-101:].astype(float)
#GDP = (GDP[:,1:]-GDP[:,:-1])/GDP[:,:-1]*1e2
#units, periods = GDP.shape


#specify the number of test and train periods
test_periods = 1
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

#HR+VT+Abadie+MC restricted

#to calculate correlation of predictions by different methods
Epsilon = np.zeros((units,test_periods,4))

#outcome matrix for training to obtain ensemble weights
Y = GDP[:,:-test_periods]

#matrix of RMSE for ensemble weighting
RMSE_weight = np.zeros((units,train_periods,4))

#record the regularization strengths obtained from cross validation
HR = np.zeros((units,2))
VT = np.zeros((units,2))

#for each unit, obtain weights from train periods first, then 
#apply individual and ensemble methods to test periods

#train for training periods
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
    
    #record predictions of individual methods and actual outcome for 
    #regression ensemble weighting
    individual_predictions = np.zeros((train_periods,4))
    actual_outcomes = np.zeros(train_periods,)
    
    for train_period in range(train_periods):
        
        #shift current train period to the last column
        Y_shift = np.roll(Y0,-train_period-1,1)
        actual_outcomes[train_period] = Y_shift[-1,-1]
        
        #obtain RMSE for ensemble weighting
        
        #METHOD-1 MC
        individual_predictions[train_period,0] = estimate.MC(Y_shift)
        
        #METHOD-2 HR
        individual_predictions[train_period,1],_,_ = estimate.HR(Y_shift,a=HR_alpha,b=HR_l1_ratio)
        
        #METHOD-3 VT
        individual_predictions[train_period,2],_,_ = estimate.VT(Y_shift,a=VT_alpha,b=VT_l1_ratio)
        
        #METHOD-4 Abadie
        individual_predictions[train_period,3] = estimate.Abadie(Y_shift)
        
        RMSE_weight[current_unit,train_period,:] = np.absolute(individual_predictions[train_period,:]-actual_outcomes[train_period])
        
    #obtain ensemble weights for regression ensemble methods
    
    #HR+VT restricted
    ensem1_weights[current_unit,:] = estimate.restricted_regression(individual_predictions[:,1:3],actual_outcomes)

    #HR+VT+MC restricted
    ensem2_weights[current_unit,:] = estimate.restricted_regression(individual_predictions[:,0:3],actual_outcomes)

    #HR+VT+Abadie+MC restricted
    ensem3_weights[current_unit,:] = estimate.restricted_regression(individual_predictions[:,0:],actual_outcomes)

    #HR+VT+Abadie restricted
    ensem4_weights[current_unit,:] = estimate.restricted_regression(individual_predictions[:,1:],actual_outcomes)
 
#obtain inverse-RMSE ensemble weighting
#RMSE_weight = np.squeeze(np.sum(np.square(RMSE_weight),1))
#RMSE_weight = np.sqrt(RMSE_weight)

#inv_RMSE = np.reciprocal(RMSE_weight)
#ensem2_weights = (inv_RMSE.T/np.sum(inv_RMSE,1)).T


#predict for test periods
for test_period in range(test_periods):
    
    #select outcomes up to the current (to be predicted) period
    Y = GDP[:,:periods-test_periods+test_period+1]
    
    for current_unit in [49]:
    
        #shift current unit to the last row
        Y0 = np.roll(Y,-current_unit-1,0)
        
        #record predictions of individual methods
        individual_predictions = np.zeros(4,)
        
        #compare with actual outcome 
        actual_outcome = Y0[-1,-1]
        
        #METHOD-1 MC
        individual_predictions[0] = estimate.MC(Y0)
        
        #METHOD-2 HR
        #individual_predictions[1],_,_ = estimate.HR(Y0,cv=True)
        individual_predictions[1],coef_HR,intercept_HR = estimate.HR(Y0,a=HR[current_unit,0],b=HR[current_unit,1])
        
        #METHOD-3 VT
        #individual_predictions[2],_,_ = estimate.SC(Y0,cv=True)
        individual_predictions[2],coef_VT,intercept_VT = estimate.VT(Y0,a=VT[current_unit,0],b=VT[current_unit,1])
        
        #METHOD-4 Abadie
        individual_predictions[3] = estimate.Abadie(Y0)
        
        Epsilon[current_unit,test_period,:] = individual_predictions-actual_outcome
        
        RMSE_test[current_unit,test_period,0:4] = np.absolute(individual_predictions-actual_outcome)
        
        #ensemble-1 HR+VT restricted
        ensem_prediction = np.dot(ensem1_weights[current_unit,:],individual_predictions[1:3])
        RMSE_test[current_unit,test_period,4] = np.absolute(ensem_prediction-actual_outcome)
        
        #ensemble-2 HR+VT+MC restricted
        ensem_prediction = np.dot(ensem2_weights[current_unit,:],individual_predictions[:3])
        RMSE_test[current_unit,test_period,5] = np.absolute(ensem_prediction-actual_outcome)
        
        #ensemble-3 HR+VT+Abadie+MC restricted
        ensem_prediction = np.dot(ensem3_weights[current_unit,:],individual_predictions[0:])
        RMSE_test[current_unit,test_period,6] = np.absolute(ensem_prediction-actual_outcome)
        
        #ensemble-4 HR+VT+Abadie restricted
        ensem_prediction = np.dot(ensem4_weights[current_unit,:],individual_predictions[1:])
        RMSE_test[current_unit,test_period,7] = np.absolute(ensem_prediction-actual_outcome)       

#calculate RMSE by averaging over test periods, i.e. axis 1        
RMSE_final = np.squeeze(np.sqrt(np.sum(np.square(RMSE_test),1)/test_periods))
RMSE_final = (RMSE_final.T/np.std(GDP,axis=1)).T
overall = np.sqrt(np.sum(np.square(RMSE_final),0)/units)
med = np.median(RMSE_final,0);

RMSE_output = np.concatenate((RMSE_final,np.reshape(overall,(1,-1)),np.reshape(med,(1,-1))),axis=0)

#calculate correlation and covariance of individual methods
cov_input = np.reshape(Epsilon.transpose(),(4,-1))

covariance_matrix = np.cov(cov_input)
correlation_matrix = np.corrcoef(cov_input)

index1 = ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','mean','median']
columns1 = ['MC','HR','VT','Abadie','HR+VT','MC+HR+VT','MC+HR+VT+Abadie','HR+VT+Abadie']
RMSE_print = pd.DataFrame(np.around(RMSE_output,5),index=index1,columns=columns1)  

index2 = ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
columns2 = ['MC','HR','VT','Abadie']

corr_print = pd.DataFrame(correlation_matrix,index=columns2,columns=columns2)
ensem1_print = pd.DataFrame(ensem1_weights,index=index2,columns=['HR','VT'])
ensem2_print = pd.DataFrame(ensem2_weights,index=index2,columns=['MC','HR','VT'])
ensem3_print = pd.DataFrame(ensem3_weights,index=index2,columns=['MC','HR','VT','Abadie'])
ensem4_print = pd.DataFrame(ensem4_weights,index=index2,columns=['HR','VT','Abadie'])
    
#RMSE_print.to_csv('RMSE.csv')
#corr_print.to_csv('corr.csv')
#ensem1_print.to_csv('ensem1.csv')
#ensem2_print.to_csv('ensem2.csv')
#ensem3_print.to_csv('ensem3.csv')
#ensem4_print.to_csv('ensem4.csv')

index3 = ['AL','AK','AZ','AR','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
HR_print = pd.DataFrame(np.concatenate((coef_HR,[intercept_HR]),axis=0))
HR_print.to_csv('HR.csv')
VT_print = pd.DataFrame(np.concatenate((coef_VT,[intercept_VT]),axis=0))
VT_print.to_csv('VT.csv')