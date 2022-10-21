data = importdata('yogurt_2018.txt');

%Q1
ave_price = zeros(3,3);

for i = 1:3
I = find(data(:,2)==i);

ave_price(i,:) = mean(data(I,3:5),1);

end

%Q2
beta = -0.04;
alpha_2 = 0.5;
alpha_3 = -1;

utilities = zeros(size(data(:,3:5)));

utilities(:,1) = data(:,3)*beta;
utilities(:,2) = data(:,4)*beta+alpha_2;
utilities(:,3) = data(:,5)*beta+alpha_3;

exp_util = exp(utilities);

samples = size(data,1);

I = (1:samples).';
k = sub2ind(size(utilities), I, data(:,2));
C = utilities(k);

log_likelihood = sum(C-log(sum(exp_util,2)));

%Q3
count_2 = size(find(data(:,2)==2),1);
count_3 = size(find(data(:,2)==3),1);

del_a2 = count_2 - sum(exp_util(:,2)./(sum(exp_util,2)));
del_a3 = count_3 - sum(exp_util(:,3)./(sum(exp_util,2)));
prices = data(:,3:5);
del_beta = sum(prices(k))-sum(sum(exp_util.*prices,2)./sum(exp_util,2));

%Q4
del2_a22 = sum(power(exp_util(:,2),2)./power(sum(exp_util,2),2)) - sum(exp_util(:,2)./(sum(exp_util,2)));
del2_a33 = sum(power(exp_util(:,3),2)./power(sum(exp_util,2),2)) - sum(exp_util(:,3)./(sum(exp_util,2)));
del2_a23 = sum((exp_util(:,3)).*(exp_util(:,2))./power(sum(exp_util,2),2));
del2_a2beta = sum(exp_util(:,2).*sum(exp_util.*prices,2)./power(sum(exp_util,2),2)) - sum(prices(:,2).*exp_util(:,2)./(sum(exp_util,2)));
del2_a3beta = sum(exp_util(:,3).*sum(exp_util.*prices,2)./power(sum(exp_util,2),2)) - sum(prices(:,3).*exp_util(:,3)./(sum(exp_util,2)));
del2_betabeta = sum(power(sum(exp_util.*prices,2),2)./power(sum(exp_util,2),2)) - sum(sum(exp_util.*power(prices,2),2)./(sum(exp_util,2)));


%Q5

runs = 10;

newton_raphson = zeros(runs,3);

newton_raphson(1,1) = alpha_2;
newton_raphson(1,2) = alpha_3;
newton_raphson(1,3) = beta;

for i = 1:runs-1
 
beta = newton_raphson(i,3);
alpha_2 = newton_raphson(i,1);
alpha_3 = newton_raphson(i,2);

utilities = zeros(size(data(:,3:5)));

utilities(:,1) = data(:,3)*beta;
utilities(:,2) = data(:,4)*beta+alpha_2;
utilities(:,3) = data(:,5)*beta+alpha_3;

exp_util = exp(utilities);

samples = size(data,1);

I = (1:samples).';
k = sub2ind(size(utilities), I, data(:,2));
C = utilities(k);

log_likelihood = sum(C-log(sum(exp_util,2)));

%first derivative
count_2 = size(find(data(:,2)==2),1);
count_3 = size(find(data(:,2)==3),1);

del_a2 = count_2 - sum(exp_util(:,2)./(sum(exp_util,2)));
del_a3 = count_3 - sum(exp_util(:,3)./(sum(exp_util,2)));
prices = data(:,3:5);
del_beta = sum(prices(k))-sum(sum(exp_util.*prices,2)./sum(exp_util,2));

%second derivative
del2_a22 = sum(power(exp_util(:,2),2)./power(sum(exp_util,2),2)) - sum(exp_util(:,2)./(sum(exp_util,2)));
del2_a33 = sum(power(exp_util(:,3),2)./power(sum(exp_util,2),2)) - sum(exp_util(:,3)./(sum(exp_util,2)));
del2_a23 = sum((exp_util(:,3)).*(exp_util(:,2))./power(sum(exp_util,2),2));
del2_a2beta = sum(exp_util(:,2).*sum(exp_util.*prices,2)./power(sum(exp_util,2),2)) - sum(prices(:,2).*exp_util(:,2)./(sum(exp_util,2)));
del2_a3beta = sum(exp_util(:,3).*sum(exp_util.*prices,2)./power(sum(exp_util,2),2)) - sum(prices(:,3).*exp_util(:,3)./(sum(exp_util,2)));
del2_betabeta = sum(power(sum(exp_util.*prices,2),2)./power(sum(exp_util,2),2)) - sum(sum(exp_util.*power(prices,2),2)./(sum(exp_util,2)));

Hessian = zeros(3,3);
Hessian(1,1) = del2_a22;
Hessian(2,2) = del2_a33;
Hessian(3,3) = del2_betabeta;
Hessian(1,2) = del2_a23;
Hessian(2,1) = del2_a23;
Hessian(1,3) = del2_a2beta;
Hessian(3,1) = del2_a2beta;
Hessian(2,3) = del2_a3beta;
Hessian(3,2) = del2_a3beta;

gradient = zeros(3,1);

gradient(1,1) = del_a2;
gradient(2,1) = del_a3;
gradient(3,1) = del_beta;

update = (inv(Hessian))*gradient;

newton_raphson(i+1,1) = alpha_2-update(1,1);
newton_raphson(i+1,2) = alpha_3-update(2,1);
newton_raphson(i+1,3) = beta-update(3,1);

max(gradient)

end

%Q7
log_likelihood = zeros(100,1);

for l = 1:100
c = l*0.2/100;
beta_L = beta-c;
beta_H = beta+c;

utilities_L = zeros(size(data(:,3:5)));
utilities_H = zeros(size(data(:,3:5)));

utilities_L(:,1) = data(:,3)*beta_L;
utilities_L(:,2) = data(:,4)*beta_L+alpha_2;
utilities_L(:,3) = data(:,5)*beta_L+alpha_3;

utilities_H(:,1) = data(:,3)*beta_H;
utilities_H(:,2) = data(:,4)*beta_H+alpha_2;
utilities_H(:,3) = data(:,5)*beta_H+alpha_3;

exp_util_L = exp(utilities_L);
exp_util_H = exp(utilities_H);

indiv_likelihood_L = exp_util_L(k)./sum(exp_util_L,2);
indiv_likelihood_H = exp_util_H(k)./sum(exp_util_H,2);

for i = 1:429
    consumer_purchases = find(data(:,1)==i);
    
    consumer_likelihood = 0.4*prod(indiv_likelihood_L(consumer_purchases))+0.6*prod(indiv_likelihood_H(consumer_purchases));
    
    log_likelihood(l) = log_likelihood(l) + log(consumer_likelihood);
end

end
x = 0.002:0.002:0.2;    
plot(x,log_likelihood)
title('log likelihood as a function of c')

%Q8

rate = 0.0001;

choice_prices = prices(k);

parameters = zeros(101,4);

parameters(1,1) = 0.9;
parameters(1,2) = -0.7;
parameters(1,3) = -0.05;
parameters(1,4) = -0.03;

for iteration = 1:1000
    
beta_L = parameters(iteration,3);
beta_H = parameters(iteration,4);
alpha_2 = parameters(iteration,1);
alpha_3 = parameters(iteration,2);

utilities_L = zeros(size(data(:,3:5)));
utilities_H = zeros(size(data(:,3:5)));

utilities_L(:,1) = data(:,3)*beta_L;
utilities_L(:,2) = data(:,4)*beta_L+alpha_2;
utilities_L(:,3) = data(:,5)*beta_L+alpha_3;

utilities_H(:,1) = data(:,3)*beta_H;
utilities_H(:,2) = data(:,4)*beta_H+alpha_2;
utilities_H(:,3) = data(:,5)*beta_H+alpha_3;

exp_util_L = exp(utilities_L);
exp_util_H = exp(utilities_H);
exp_util_L_choice = exp_util_L(k);
exp_util_H_choice = exp_util_H(k);

%E-step

posterior = zeros(429,1);

for i = 1:429
    consumer_purchases = find(data(:,1)==i);
    like_L = prod(exp_util_L_choice(consumer_purchases)./sum(exp_util_L(consumer_purchases,:),2));
    like_H = prod(exp_util_H_choice(consumer_purchases)./sum(exp_util_H(consumer_purchases,:),2));
    posterior(i) = 0.4*like_L/(0.4*like_L+0.6*like_H);
end



%M-step

%vector of posterior probabilities for each purchase
post = zeros(samples,1);
for i = 1:samples
consumer_purchases = find(data(:,1)==i);
post(i,1) = posterior(data(i,1));
end

cvx_begin
    variable alpha_2
    variable alpha_3
    variable beta_L
    variable beta_H
    
    utilities_L_1 = data(:,3)*beta_L;
    utilities_L_2 = data(:,4)*beta_L+alpha_2;
    utilities_L_3 = data(:,5)*beta_L+alpha_3;

    utilities_H_1 = data(:,3)*beta_H;
    utilities_H_2 = data(:,4)*beta_H+alpha_2;
    utilities_H_3 = data(:,5)*beta_H+alpha_3;

    exp_util_L = exp([utilities_L_1 utilities_L_2 utilities_L_3]);
    exp_util_H = exp([utilities_H_1 utilities_H_2 utilities_H_3]);
    util_L_choice = utilities_L(k);
    util_H_choice = utilities_H(k);
    
    likelihood_L = (util_L_choice)-log(sum(exp_util_L,2));
    likelihood_H = (util_H_choice)-log(sum(exp_util_H,2));
    log_likelihood = sum(post.*likelihood_L+(1-post).*likelihood_H);

    maximize (log_likelihood)
    
cvx_end
    
    parameters(iteration+1,3) = beta_L;
    parameters(iteration+1,4) = beta_H;
    parameters(iteration+1,1) = alpha_2;
    parameters(iteration+1,2) = alpha_3;
end
