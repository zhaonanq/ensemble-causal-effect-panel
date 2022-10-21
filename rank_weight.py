import numpy as np
import individual_methods_rank as estimate
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#read in data
data = pd.read_csv('GDP_data.csv')

ranks = []
hr_weights = []
vt_weights = []

#data type
data_type = 'GDP_100_'

#select the last 100 periods and then scale data by 1e-8
GDP = data.values[:,-100:].astype(float)*1e-8
#GDP = np.log(data.values[:,-100:].astype(float))
#units, periods = GDP.shape

#GDP growth rate
#GDP = data.values[:,-26:].astype(float)
#GDP = (GDP[:,1:]-GDP[:,:-1])/GDP[:,:-1]*1e2
units, periods = GDP.shape

#specify the number of test and train periods
test_periods = 10
train_periods = periods-test_periods

#record the regularization strengths obtained from cross validation
HR = np.zeros((units,2))
VT = np.zeros((units,2))

#for each unit, obtain weights from train periods first, then 
#apply individual and ensemble methods to test periods
Y = GDP[:,:-test_periods]
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
    

#predict for test periods
for test_period in range(test_periods):
    
    #select outcomes up to the current (to be predicted) period
    Y = GDP[:,:periods-test_periods+test_period+1]
    
    for current_unit in range(units):
    
        #shift current unit to the last row
        Y0 = np.roll(Y,-current_unit-1,0)
        
        #METHOD-1 MC
        ranks.append(estimate.MC_fe(Y0))
        
        #METHOD-2 HR
        hr_weights.append(estimate.HR(Y0,a=HR[current_unit,0],b=HR[current_unit,1]))
        
        #METHOD-3 VT
        vt_weights.append(estimate.VT(Y0,a=VT[current_unit,0],b=VT[current_unit,1]))
        
        
median_print = pd.DataFrame([np.median(vt_weights),np.median(hr_weights),np.median(ranks)])
mean_print = pd.DataFrame([np.mean(vt_weights),np.mean(hr_weights),np.mean(ranks)])
 
median_print.to_csv(data_type+'median.csv')
mean_print.to_csv(data_type+'mean.csv')