import pandas as pd
import numpy as np
data = pd.read_csv('raw_bike_data.csv',usecols=['Start_Time','Start_Station_Name','Holiday','Date','Count'])
Dates = data.Date.unique()
Stations = data.Start_Station_Name.unique()
outcome_matrix = np.zeros((620,1))
for i in range(len(Dates)):
    to_append = np.zeros((620,24))
    for j in range(620):
        to_append[j,:] = data.loc[(data['Date'] == Dates[i]) & (data['Start_Station_Name'] == Stations[j])]['Count'].values
    outcome_matrix = np.concatenate((outcome_matrix,to_append),axis=1)
output = pd.DataFrame(data=outcome_matrix,index=Stations)
output.to_csv('bike_outcomes.csv')