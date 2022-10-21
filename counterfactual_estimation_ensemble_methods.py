import numpy as np
import individual_methods_sherlock as estimate
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#read in data
data = pd.read_csv('GDP_data.csv')

#data type
data_type = 'GDP_log_10_'

#select the last 100 periods and then scale data by 1e-8
#GDP = data.values[:,-25:].astype(float)*1e-8
GDP = np.log(data.values[:,-10:].astype(float))
units, periods = GDP.shape

#GDP growth rate
#GDP = data.values[:,-11:].astype(float)
#GDP = (GDP[:,1:]-GDP[:,:-1])/GDP[:,:-1]*1e2
#units, periods = GDP.shape

#specify the number of test and train periods
test_periods = 4
train_periods = periods-test_periods

#record RMSE of predictions by 8 methods
RMSE_test = np.zeros((units,test_periods,10))

#Vertical cross validation, different for each test period

#ensemble weights for generalized stacking
#HR+VT+MC unrestricted, vertical cross validation
vt_unres_weights = np.zeros((units,3))
vt_unres_intercepts = np.zeros(units,)

#HR+VT+MC restricted, vertical cross validation
vt_res_weights = np.zeros((units,3))

#random forest regressor
vt_RF_regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)

#Horizontal cross validation, same for each unit across all periods

#ensemble weights for generalized stacking
#HR+VT+MC unrestricted, horizontal cross validation
hr_unres_weights = np.zeros((units,3))
hr_unres_intercepts = np.zeros(units,)

#HR+VT+MC restricted, horizontal cross validation
hr_res_weights = np.zeros((units,3))

#random forest regressor, list of regressors one for each unit
hr_RF = []

#to calculate correlation of predictions by different methods
Epsilon = np.zeros((units,test_periods,4))

#outcome matrix for training to obtain ensemble weights for horizontal cv
Y = GDP[:,:-test_periods]

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
        individual_predictions[train_period,0] = estimate.MC_fe(Y_shift)
        
        #METHOD-2 HR
        individual_predictions[train_period,1],_,_ = estimate.HR(Y_shift,a=HR_alpha,b=HR_l1_ratio)
        
        #METHOD-3 VT
        individual_predictions[train_period,2],_,_ = estimate.VT(Y_shift,a=VT_alpha,b=VT_l1_ratio)
        
        #METHOD-4 neural net
        individual_predictions[train_period,3] = estimate.Abadie(Y_shift)
                
    #obtain ensemble weights for ensemble method, horizontal cv
    
    #HR+VT+MC unrestricted
    reg = LinearRegression().fit(individual_predictions[:,:3], actual_outcomes)
    hr_unres_weights[current_unit,:] = reg.coef_
    hr_unres_intercepts[current_unit] = reg.intercept_

    #HR+VT+MC restricted
    hr_res_weights[current_unit,:] = estimate.restricted_regression(individual_predictions[:,:3],actual_outcomes)

    #random forest 
    
    #grid search
    param_grid = {
    'bootstrap': [True,False],
    'max_depth': [5,10,50,100],
    'n_estimators': [5,10,50,100]
    }
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

    grid_search.fit(individual_predictions[:,:3], actual_outcomes)
    hr_RF_regr = grid_search.best_estimator_
    hr_RF.append(hr_RF_regr)
    
#record sample predictions and actual outcome for desired unit for the last period
desired_unit = 49
results = np.zeros(11,)

#predict for test periods
for test_period in range(test_periods):
    
    #select outcomes up to the current (to be predicted) period
    Y = GDP[:,:periods-test_periods+test_period+1]
    
    #record predictions of individual methods and actual outcome for 
    #regression ensemble weighting with vertical cv
    individual_predictions = np.zeros((units,4))
    actual_outcomes = np.zeros(units,)
    
    for current_unit in range(units):
    
        #shift current unit to the last row
        Y0 = np.roll(Y,-current_unit-1,0)
        
        #compare with actual outcome 
        actual_outcome = Y0[-1,-1]
        actual_outcomes[current_unit] = actual_outcome
        
        #METHOD-1 MC
        individual_predictions[current_unit,0] = estimate.MC_fe(Y0)
        
        #METHOD-2 HR
        individual_predictions[current_unit,1],coef_HR,intercept_HR = estimate.HR(Y0,a=HR[current_unit,0],b=HR[current_unit,1])
        
        #METHOD-3 VT
        individual_predictions[current_unit,2],coef_VT,intercept_VT = estimate.VT(Y0,a=VT[current_unit,0],b=VT[current_unit,1])
        
        #METHOD-4 neural net
        individual_predictions[current_unit,3] = estimate.Abadie(Y0)
        
        #HR+VT+MC unrestricted, hr cv
        ensem_prediction_1 = np.dot(hr_unres_weights[current_unit,:],individual_predictions[current_unit,0:3])+hr_unres_intercepts[current_unit]
        RMSE_test[current_unit,test_period,4] = np.absolute(ensem_prediction_1-actual_outcome)
        
        #HR+VT+MC restricted, hr cv
        ensem_prediction_2 = np.dot(hr_res_weights[current_unit,:],individual_predictions[current_unit,0:3])
        RMSE_test[current_unit,test_period,5] = np.absolute(ensem_prediction_2-actual_outcome)
            
        #random forest, hr cv  
        ensem_prediction_3 = hr_RF[current_unit].predict(individual_predictions[current_unit,0:3].reshape(1,-1))
        RMSE_test[current_unit,test_period,6] = np.absolute(ensem_prediction_3-actual_outcome)

        #record the predictions and actual outcomes for a desired state in the last period
        if current_unit == desired_unit and test_period == test_periods-1:
            HR_output = np.concatenate((coef_HR,[intercept_HR]),axis=0)
            VT_output = np.concatenate((coef_VT,[intercept_VT]),axis=0)
            results[0:4] = individual_predictions[current_unit,:]
            results[4] = ensem_prediction_1
            results[5] = ensem_prediction_2
            results[6] = ensem_prediction_3
            results[-1] = actual_outcome

    Epsilon[:,test_period,:] = (individual_predictions.T-actual_outcomes).T
        
    RMSE_test[:,test_period,0:4] = np.absolute(individual_predictions.T-actual_outcomes).T

    #obtain ensemble weights and predictions for vertical cross-validation
    
    for current_unit in range(units):
        
        mask = (np.arange(units) != current_unit)
        
        cv_predictions = individual_predictions[mask,:]
        cv_outcomes = actual_outcomes[mask]
        
        actual_outcome = actual_outcomes[current_unit]
        
        #HR+VT+MC unrestricted, vt cv
        reg = LinearRegression().fit(cv_predictions[:,:3], cv_outcomes)
        vt_unres_weights[current_unit,:] = reg.coef_
        vt_unres_intercepts[current_unit] = reg.intercept_
        
        ensem_prediction_1 = np.dot(reg.coef_,individual_predictions[current_unit,:3])+reg.intercept_
        RMSE_test[current_unit,test_period,7] = np.absolute(ensem_prediction_1-actual_outcome)

        #HR+VT+MC restricted, vt cv
        vt_res_weights[current_unit,:] = estimate.restricted_regression(cv_predictions[:,:3],cv_outcomes)
        ensem_prediction_2 = np.dot(vt_res_weights[current_unit,:],individual_predictions[current_unit,:3])
        RMSE_test[current_unit,test_period,8] = np.absolute(ensem_prediction_2-actual_outcome)

        #random forest, vt cv
        #grid search
        param_grid = {
            'bootstrap': [True,False],
            'max_depth': [5,10,50,100],
            'n_estimators': [5,10,50,100]
            }
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(cv_predictions[:,:3], cv_outcomes)
        vt_RF_regr = grid_search.best_estimator_
  
        ensem_prediction_3 = vt_RF_regr.predict(individual_predictions[current_unit,0:3].reshape(1,-1))
        RMSE_test[current_unit,test_period,9] = np.absolute(ensem_prediction_3-actual_outcome)
        
        #record the predictions and actual outcomes for a desired state in the last period
        if current_unit == desired_unit and test_period == test_periods-1:
            results[7] = ensem_prediction_1
            results[8] = ensem_prediction_2
            results[9] = ensem_prediction_3
            
#calculate RMSE by averaging over test periods, i.e. axis 1        
RMSE_final = np.squeeze(np.sqrt(np.sum(np.square(RMSE_test),1)/test_periods))
RMSE_final = (RMSE_final.T/np.std(GDP,axis=1)).T
overall = np.sqrt(np.sum(np.square(RMSE_final),0)/units)
#med = np.median(RMSE_final,0);

RMSE_output = np.concatenate((RMSE_final,np.reshape(overall,(1,-1))),axis=0)

#calculate correlation and covariance of individual methods
cov_input = np.reshape(Epsilon.transpose(),(4,-1))

covariance_matrix = np.cov(cov_input)
correlation_matrix = np.corrcoef(cov_input)

index1 = ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','mean']
columns1 = ['MC','HR','VT','Abadie','unres_hr','res_hr','RF_hr','unres_vt','res_vt','RF_vt']
RMSE_print = pd.DataFrame(np.around(RMSE_output,3),index=index1,columns=columns1)  

index2 = ['AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
columns2 = ['MC','HR','VT','Abadie']

corr_print = pd.DataFrame(correlation_matrix,index=columns2,columns=columns2)

vt_unres_print = pd.DataFrame(np.concatenate((vt_unres_weights,np.reshape(vt_unres_intercepts,(-1,1))),1),index=index2,columns=['MC','HR','VT','intercept'])

vt_res_print = pd.DataFrame(vt_res_weights,index=index2,columns=['MC','HR','VT'])

hr_unres_print = pd.DataFrame(np.concatenate((hr_unres_weights,np.reshape(hr_unres_intercepts,(-1,1))),1),index=index2,columns=['MC','HR','VT','intercept'])

hr_res_print = pd.DataFrame(hr_res_weights,index=index2,columns=['MC','HR','VT'])

RMSE_print.to_csv(data_type+'RMSE.csv')
corr_print.to_csv(data_type+'corr.csv')
vt_unres_print.to_csv(data_type+'vt_unres.csv')
vt_res_print.to_csv(data_type+'vt_res.csv')
hr_unres_print.to_csv(data_type+'hr_unres.csv')
hr_res_print.to_csv(data_type+'hr_res.csv')

HR_print = pd.DataFrame(HR_output)
HR_print.to_csv(data_type+'HR_weights.csv')
VT_print = pd.DataFrame(VT_output)
VT_print.to_csv(data_type+'VT_weights.csv')
result_print = pd.DataFrame(results)
result_print.to_csv(data_type+'predictions.csv')