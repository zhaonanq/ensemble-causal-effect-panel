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
    
RMSE_weight = zeros(51,50,4);
    
%training stage

Y0 = (income(:,1:current_period)).';

M = covariates(:,:,1:current_period);

for shift = 1:51
    
shift

Y_shift = circshift(Y0,[0,-shift]);

M_shift = circshift(M,[0,-shift,0]);

%obtain RMSE for ensemble weighting

cross_valid = Y_shift(:,1:end-1);
M_cv = M_shift(:,1:end-1,:);

Pred = zeros(50,4);
Actual = zeros(50,1);

for left_out = 1:50
    
    cv_shift = circshift(cross_valid,[0,left_out]);
    
    %METHOD-1 MC
    [predicted_outcomes, actual_outcomes] = MC(cv_shift,1);
    RMSE_weight(shift,left_out,1) = sqrt(immse(predicted_outcomes,actual_outcomes));
    Pred(left_out,1) = predicted_outcomes;
    
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
    
    Actual(left_out,1) = actual_outcomes;
end

cvx_begin
    variable v(4,1) nonnegative

    square_sum = sum_square((sum(Pred*v,2)-Actual));

    minimize (square_sum)
    subject to
        sum(v) == 1;
cvx_end

ensem3_weights(shift,:) = v.';

cvx_begin
    variable w(4,1)
    variable nu

    square_sum = sum_square((sum(Pred*w,2)+nu*ones(50,1)-Actual));

    minimize (square_sum)
    
cvx_end

ensem4_weights(shift,:) = w.';
ensem4_intercepts(shift,:) = nu;

end

%obtain weights

RMSE_weight = squeeze(sum(RMSE_weight.^2,2))/50;

RMSE_weight = sqrt(RMSE_weight);

inv_RMSE = RMSE_weight.^(-1);

ensem2_weights = inv_RMSE./(sum(inv_RMSE,2));

ensem1_weights(:,:) = 1/4;

weights2(:,:,current_period) = ensem2_weights;
weights3(:,:,current_period) = ensem3_weights;
weights4(:,:,current_period) = ensem4_weights;

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




save(variables)
