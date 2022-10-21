function [predicted_outcomes, actual_outcomes] = DID(outcome_matrix,prediction_period)

[periods,units] = size(outcome_matrix);

actual_outcomes = outcome_matrix(end,end);

cvx_begin quiet

variable a(periods,1)
variable c(1,units)

fixed_effects = ones(units,periods)*diag(a) + diag(c)*ones(units,periods);
diff = reshape(outcome_matrix.', [], 1)- reshape(fixed_effects, [], 1);
diff = diff(1:end-1);
square_loss = sum(diff.^2);

minimize (square_loss);

cvx_end

predicted_outcomes = a(periods,1)+c(1,units);


end