function [predicted_outcomes, actual_outcomes] = SC_abadie(outcome_matrix,prediction_period)

[periods,units] = size(outcome_matrix);

Y0_shift = outcome_matrix(:,1:units-1);
Y1_shift = outcome_matrix(:,end);

Z0_shift = Y0_shift(1:end-prediction_period,:);

Z1_shift = Y1_shift(1:end-prediction_period,:);

lambda = 0.1;
alpha = 0.1;

cvx_begin quiet
variable w(units-1,1) nonnegative

square_sum = sum_square((sum(Z0_shift*w,2)-Z1_shift));

%regularization = (1-alpha)/2*sum_square(w)+alpha*norm(w,1);

minimize (square_sum)

subject to 
    sum(w) == 1;
cvx_end


predicted_outcomes = Y0_shift(end-prediction_period+1:end,:)*w;
actual_outcomes = Y1_shift(end-prediction_period+1:end,:);

end