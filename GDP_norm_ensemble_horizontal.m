cd cvx
cvx_setup
cd ..

%regularization strength
reg_rate = 1e-2;

data = csvread('GDP_data.csv',1,0);

income = data(:,end-49:end)*1e-8;

test_periods = 5;

[units, periods] = size(income);

train_periods = periods-test_periods;

income = diag(1./std(income,0,2))*(income-mean(income,2));

RMSE_test = zeros(units,9,test_periods);

ensem1_weights = zeros(units,5);
ensem2_weights = zeros(units,5);
ensem3_weights = zeros(units,5);
ensem4_weights = zeros(units,5);
ensem4_intercepts = zeros(units,1);

%to calculate correlation of predictions by different methods
Epsilon = zeros(units,test_periods,5);

%outcome matrix for training
Y = (income(:,1:train_periods)).';

%matrix of RMSE for weighting
RMSE_weight = zeros(units,periods-test_periods,5);

%obtain weight for ensemble methods
for current_unit = 1:units
   
Y0 = circshift(Y,[0,-current_unit]);

%choose a few random units to perform regularization selection for MC
%method

num_MC = 5;

indices = randi(train_periods,num_MC,1);
errors = zeros(num_MC,3);
regularization = [1e-1 1e-2 1e-3];

for i = 1:num_MC
    
    cv_shift = circshift(Y0,[-indices(i),0]);
    
    for j = 1:3
        
    %MC
    [predicted_outcomes, actual_outcomes, matrix_rank] = MC(cv_shift,1,regularization(j));
    
    errors(i,j) = sqrt(immse(predicted_outcomes,actual_outcomes));
    
    end
    
end

reg_rmse = sqrt(sum(errors.^2,1)/num_MC);

[val,reg_index] = min(reg_rmse);

reg_rate = regularization(reg_index);
%}

Pred = zeros(train_periods,5);
Actual = zeros(train_periods,1);

for shift = 1:train_periods
    
shift 

Y_shift = circshift(Y0,[-shift,0]);

%obtain RMSE for ensemble weighting

%METHOD-1 MC
[predicted_outcomes, actual_outcomes, matrix_rank] = MC(Y_shift,1,reg_rate);
matrix_rank
RMSE_weight(current_unit,shift,1) = sqrt(immse(predicted_outcomes,actual_outcomes));
Pred(shift,1) = predicted_outcomes;

%METHOD-2 SC_no
[predicted_outcomes, actual_outcomes] = SC_no(Y_shift,1);
RMSE_weight(current_unit,shift,2) = sqrt(immse(predicted_outcomes,actual_outcomes));
Pred(shift,2) = predicted_outcomes;
    
%METHOD-3 HR_no
[predicted_outcomes, actual_outcomes] = HR_no(Y_shift,1);
RMSE_weight(current_unit,shift,3) = sqrt(immse(predicted_outcomes,actual_outcomes));
Pred(shift,3) = predicted_outcomes;
    
%METHOD-4 SC_abadie
[predicted_outcomes, actual_outcomes] = SC_abadie(Y_shift,1);
RMSE_weight(current_unit,shift,4) = sqrt(immse(predicted_outcomes,actual_outcomes));
Pred(shift,4) = predicted_outcomes;
    
%METHOD-5 DID
[predicted_outcomes, actual_outcomes] = DID(Y_shift,1);
RMSE_weight(current_unit,shift,5) = sqrt(immse(predicted_outcomes,actual_outcomes));
Pred(shift,5) = predicted_outcomes;
    
Actual(shift,1) = actual_outcomes;
end

cvx_begin quiet
    variable v(5,1) nonnegative

    square_sum = sum_square((sum(Pred*v,2)-Actual));

    minimize (square_sum)
    subject to
        sum(v) == 1;
cvx_end

ensem3_weights(current_unit,:) = v.';

cvx_begin quiet
    variable w(5,1)
    variable nu

    square_sum = sum_square((sum(Pred*w,2)+nu*ones(train_periods,1)-Actual));

    minimize (square_sum)
    
cvx_end

ensem4_weights(current_unit,:) = w.';
ensem4_intercepts(current_unit,:) = nu;

end

save("GDP_norm_horizontal")

%obtain weights

RMSE_weight = squeeze(sum(RMSE_weight.^2,2));

RMSE_weight = sqrt(RMSE_weight);

inv_RMSE = RMSE_weight.^(-1);

ensem2_weights = inv_RMSE./(sum(inv_RMSE,2));

ensem1_weights(:,:) = 1/5;

%testing stage

for current_period = train_periods+1:periods
    
current_period

Y0 = income(:,1:current_period).';

for shift = 1:units
    
unit_outcome = zeros(1,5);

Y_shift = circshift(Y0,[0,-shift]);

%METHOD-1 MC

[predicted_outcomes, actual_outcomes,matrix_rank] = MC(Y_shift,1,reg_rate);

unit_outcome(:,1) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period-periods+test_periods,1) = epsilon;

RMSE_test(shift,1,current_period-(periods-test_periods)) = abs(epsilon);

%METHOD-2 SC-EN 

[predicted_outcomes, actual_outcomes] = SC_no(Y_shift,1);

unit_outcome(:,2) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period-periods+test_periods,2) = epsilon;

RMSE_test(shift,2,current_period-(periods-test_periods)) = abs(epsilon);

% METHOD-3 horizontal regression 

[predicted_outcomes, actual_outcomes] = HR_no(Y_shift,1);

unit_outcome(:,3) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period-periods+test_periods,3) = epsilon;

RMSE_test(shift,3,current_period-(periods-test_periods)) = abs(epsilon);

%METHOD-4 SC-abadie

[predicted_outcomes, actual_outcomes] = SC_abadie(Y_shift,1);

unit_outcome(:,4) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period-periods+test_periods,4) = epsilon;

RMSE_test(shift,4,current_period-(periods-test_periods)) = abs(epsilon);

%METHOD-5 DID

[predicted_outcomes, actual_outcomes] = DID(Y_shift,1);

unit_outcome(:,5) = predicted_outcomes;

epsilon = predicted_outcomes-actual_outcomes;

Epsilon(shift,current_period-periods+test_periods,5) = epsilon;

RMSE_test(shift,5,current_period-(periods-test_periods)) = abs(epsilon);


% METHOD-6 ensemble-1

predicted_outcomes = unit_outcome*(ensem1_weights(shift,:).');
RMSE_test(shift,6,current_period-(periods-test_periods)) = sqrt(immse(predicted_outcomes,actual_outcomes));

% METHOD-7 ensemble-2

predicted_outcomes = unit_outcome*(ensem2_weights(shift,:).');
RMSE_test(shift,7,current_period-(periods-test_periods)) = sqrt(immse(predicted_outcomes,actual_outcomes));

% METHOD-8 ensemble-3

predicted_outcomes = unit_outcome*(ensem3_weights(shift,:).');
RMSE_test(shift,8,current_period-(periods-test_periods)) = sqrt(immse(predicted_outcomes,actual_outcomes));

% METHOD-9 ensemble-4

predicted_outcomes = unit_outcome*(ensem4_weights(shift,:).')+ensem4_intercepts(shift,1);
RMSE_test(shift,9,current_period-(periods-test_periods)) = sqrt(immse(predicted_outcomes,actual_outcomes));

end
end

RMSE_final = squeeze(sqrt(sum(RMSE_test.^2,3)/test_periods));

RMSE_final = diag(1./std(income,0,2))*RMSE_final;

cov_input = reshape(Epsilon,[],5);

covariance_matrix = cov(cov_input);
correlation_matrix = corrcoef(cov_input);
overall = sqrt(sum(RMSE_final.^2,1)/test_periods);
med = median(RMSE_final,1);
statistics = [overall;med];
print = [RMSE_final;statistics];
ensem4_coef = [ensem4_weights ensem4_intercepts];

save("GDP_norm_horizontal")


