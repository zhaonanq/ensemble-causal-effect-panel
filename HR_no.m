function [predicted_outcomes, actual_outcomes] = HR_no(outcome_matrix,prediction_period)

[periods,units] = size(outcome_matrix);

Y0_shift = outcome_matrix(:,1:units-1);

Z0_shift = Y0_shift(1:end-prediction_period,:);
Z2_shift = Y0_shift(end-prediction_period+1:end,:);

lambda = 0.1;
alpha = 0.1;

cvx_begin quiet
    variable w(periods-prediction_period,prediction_period)
    variable nu(1,prediction_period)
    
    for period = 1:prediction_period
   
    square_sum = sum_square((sum(Z0_shift.'*w(:,period),2)+nu(1,period)-Z2_shift(end-period+1,:).'));

    regularization = (1-alpha)/2*sum_square(w(:,period))+alpha*norm(w(:,period),1);

    minimize (square_sum+lambda*regularization)
    
    end

cvx_end

predicted_outcomes = outcome_matrix(1:end-prediction_period,end).'*w+nu;
actual_outcomes = outcome_matrix(end-prediction_period+1:end,end);

end