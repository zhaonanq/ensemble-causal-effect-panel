function [predicted_outcomes, actual_outcomes, matrix_rank] = MC(outcome_matrix,prediction_period,reg_rate)

[periods,units] = size(outcome_matrix);

M = ones(1,units,periods);

y = reshape(outcome_matrix,[],1);

actual_outcomes = y(end-prediction_period+1:end,1);

y = y(1:end-prediction_period);

%low rank approximation using nuclear norm

cvx_begin quiet
    variable C(1,units,periods)
    y_predict = squeeze(sum(M.*C,1));
    y_predict = reshape(y_predict.',[],1);
    y_predict = y_predict(1:end-prediction_period,1);

    % nuclear norm
    C_0 = squeeze(C);

    minimize (reg_rate*(norm_nuc(C_0))+sum_square(y_predict-y)/(units*periods))
cvx_end

pred = sum(M(:,end,end-prediction_period+1:end).*C(:,end,end-prediction_period+1:end),1);

predicted_outcomes = squeeze(pred);

matrix_rank = rank(squeeze(C),1e-7);

end