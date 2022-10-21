function [predicted_outcomes, actual_outcomes] = SC_no(outcome_matrix,prediction_period)

[periods,units] = size(outcome_matrix);

Y0_shift = outcome_matrix(:,1:units-1);
Y1_shift = outcome_matrix(:,end);

Z0_shift = Y0_shift(1:end-prediction_period,:);

Z1_shift = Y1_shift(1:end-prediction_period,:);

lambda = 0.1;
alpha = 0.1;

cvx_begin quiet
variable w(units-1,1)
variable nu(1,1)

square_sum = sum_square((sum(Z0_shift*w,2)+nu-Z1_shift));

regularization = (1-alpha)/2*sum_square(w)+alpha*norm(w,1);

minimize (square_sum+lambda*regularization)
cvx_end


predicted_outcomes = Y0_shift(end-prediction_period+1:end,:)*w+nu;
actual_outcomes = Y1_shift(end-prediction_period+1:end,:);

end