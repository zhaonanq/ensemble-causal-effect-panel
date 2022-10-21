cd cvx
cvx_setup
cd ..

RMSE_train = zeros(51,4);
RMSE_test = zeros(51,8,10);

ensem1_weights = zeros(51,4);
ensem2_weights = zeros(51,4);
ensem3_weights = zeros(51,4);
ensem4_weights = zeros(51,4);
ensem4_intercepts = zeros(51,1);

weights2 = zeros(51,4,10);
weights3 = zeros(51,4,10);
weights4 = zeros(51,4,10);

data = csvread('income_data.csv');

income = data(3:3:end,:)*1e-5;

income = income(2:52,:);

population = data(2:3:end-1,:)*1e-7;

population = population(2:52,:);

growth_rate = zeros(51,30);

for period = 2:30
    growth_rate(:,period) = (income(:,period)-income(:,period-1))./income(:,period-1);
end

income = growth_rate*100;

covariates = ones(2,51,30);

covariates(2,:,:) = population;

Epsilon = zeros(51,10,4);

for current_period = 21:30
    
ensem1_weights(:,:) = 1/4;
ensem2_weights(:,:) = 1/4;
ensem3_weights(:,:) = 1/4;
ensem4_weights(:,:) = 1/4;

%testing stage

Y0 = income(:,1:current_period).';



for shift = 1:51
    
shift

unit_outcome = zeros(1,4);

Y_shift = circshift(Y0,[0,-shift]);

%METHOD-1 MC

[predicted_outcomes, actual_outcomes] = MC(Y_shift,1);

unit_outcome(:,1) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period,1) = epsilon;

RMSE_test(shift,1,current_period-20) = abs(epsilon);

%METHOD-2 SC-EN 

[predicted_outcomes, actual_outcomes] = SC_no(Y_shift,1);

unit_outcome(:,2) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period,2) = epsilon;

RMSE_test(shift,2,current_period-20) = abs(epsilon);

% METHOD-3 horizontal regression 

[predicted_outcomes, actual_outcomes] = HR_no(Y_shift,1);

unit_outcome(:,3) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period,3) = epsilon;

RMSE_test(shift,3,current_period-20) = abs(epsilon);

%METHOD-4 SC-abadie

[predicted_outcomes, actual_outcomes] = SC_abadie(Y_shift,1);

unit_outcome(:,4) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period,4) = epsilon;

RMSE_test(shift,4,current_period-20) = abs(epsilon);

% METHOD-5 ensemble-1

predicted_outcomes = unit_outcome*(ensem1_weights(shift,:).');
RMSE_test(shift,5,current_period-20) = sqrt(immse(predicted_outcomes,actual_outcomes));

% METHOD-6 ensemble-2

predicted_outcomes = unit_outcome*(ensem2_weights(shift,:).');
RMSE_test(shift,6,current_period-20) = sqrt(immse(predicted_outcomes,actual_outcomes));

% METHOD-7 ensemble-3

predicted_outcomes = unit_outcome*(ensem3_weights(shift,:).');
RMSE_test(shift,7,current_period-20) = sqrt(immse(predicted_outcomes,actual_outcomes));

% METHOD-8 ensemble-4

predicted_outcomes = unit_outcome*(ensem4_weights(shift,:).')+ensem4_intercepts(shift,1);
RMSE_test(shift,8,current_period-20) = sqrt(immse(predicted_outcomes,actual_outcomes));

end

end

RMSE_final = squeeze(sqrt(sum(RMSE_test.^2,3)/10));

cov_input = reshape(Epsilon,[],4);

covariance_matrix = cov(cov_input);
correlation_matrix = corrcoef(cov_input);


save('variables')
