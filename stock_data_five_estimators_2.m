
cd cvx
cvx_setup
cd ..


periods = 50;
units = 100;
test_periods = 5;

data = csvread('geithner_outcome.csv');
income = data(201:300,201:250)*1e2;

RMSE_train = zeros(units,5);
RMSE_test = zeros(units,9,test_periods);

ensem1_weights = zeros(units,5);
ensem2_weights = zeros(units,5);
ensem3_weights = zeros(units,5);
ensem4_weights = zeros(units,5);
ensem4_intercepts = zeros(units,1);

weights2 = zeros(units,5,test_periods);
weights3 = zeros(units,5,test_periods);
weights4 = zeros(units,5,test_periods);

covariates = ones(1,units,periods);

Epsilon = zeros(units,test_periods,5);

ranks = zeros(units, units-1, test_periods);

for current_period = periods-test_periods+1:periods
    
RMSE_weight = zeros(units,units-1,5);
    
%training stage

Y0 = (income(:,1:current_period)).';

M = covariates(:,:,1:current_period);

for shift = 1:units
    
shift

Y_shift = circshift(Y0,[0,-shift]);

M_shift = circshift(M,[0,-shift,0]);

%obtain RMSE for ensemble weighting

cross_valid = Y_shift(:,1:end-1);
M_cv = M_shift(:,1:end-1,:);

Pred = zeros(units-1,5);
Actual = zeros(units-1,1);

for left_out = 1:units-1
    
    cv_shift = circshift(cross_valid,[0,left_out]);
    
    %METHOD-1 MC
    [predicted_outcomes, actual_outcomes, matrix_rank] = MC(cv_shift,1);
   
    RMSE_weight(shift,left_out,1) = sqrt(immse(predicted_outcomes,actual_outcomes));
    Pred(left_out,1) = predicted_outcomes;
    
    ranks(shift, left_out, current_period-periods+test_periods) = matrix_rank;
    
    %METHOD-2 SC_no
    [predicted_outcomes, actual_outcomes] = SC_no(cv_shift,1);
    RMSE_weight(shift,left_out,2) = sqrt(immse(predicted_outcomes,actual_outcomes));
    Pred(left_out,2) = predicted_outcomes;
    
    %METHOD-3 HR_no
    [predicted_outcomes, actual_outcomes] = HR_no(cv_shift,1);
    RMSE_weight(shift,left_out,3) = sqrt(immse(predicted_outcomes,actual_outcomes));
    Pred(left_out,3) = predicted_outcomes;
    
    %METHOD-4 SC_abadie
    [predicted_outcomes, actual_outcomes] = SC_abadie(cv_shift,1);
    RMSE_weight(shift,left_out,4) = sqrt(immse(predicted_outcomes,actual_outcomes));
    Pred(left_out,4) = predicted_outcomes;
    
    %METHOD-5 DID
    [predicted_outcomes, actual_outcomes] = DID(cv_shift,1);
    RMSE_weight(shift,left_out,5) = sqrt(immse(predicted_outcomes,actual_outcomes));
    Pred(left_out,5) = predicted_outcomes;
    
    Actual(left_out,1) = actual_outcomes;
end

cvx_begin
    variable v(5,1) nonnegative

    square_sum = sum_square((sum(Pred*v,2)-Actual));

    minimize (square_sum)
    subject to
        sum(v) == 1;
cvx_end

ensem3_weights(shift,:) = v.';

cvx_begin
    variable w(5,1)
    variable nu

    square_sum = sum_square((sum(Pred*w,2)+nu*ones(units-1,1)-Actual));

    minimize (square_sum)
    
cvx_end

ensem4_weights(shift,:) = w.';
ensem4_intercepts(shift,:) = nu;

end

%obtain weights

RMSE_weight = squeeze(sum(RMSE_weight.^2,2))/(units-1);

RMSE_weight = sqrt(RMSE_weight);

inv_RMSE = RMSE_weight.^(-1);

ensem2_weights = inv_RMSE./(sum(inv_RMSE,2));

ensem1_weights(:,:) = 1/5;

weights2(:,:,current_period) = ensem2_weights;
weights3(:,:,current_period) = ensem3_weights;
weights4(:,:,current_period) = ensem4_weights;

%testing stage

Y0 = income(:,1:current_period).';

for shift = 1:units
    
shift

unit_outcome = zeros(1,5);

Y_shift = circshift(Y0,[0,-shift]);

%METHOD-1 MC

[predicted_outcomes, actual_outcomes,matrix_rank] = MC(Y_shift,1);

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

save('variables_five_estimators_2')

end

RMSE_final = squeeze(sqrt(sum(RMSE_test.^2,3)/test_periods));

cov_input = reshape(Epsilon,[],5);

covariance_matrix = cov(cov_input);
correlation_matrix = corrcoef(cov_input);


