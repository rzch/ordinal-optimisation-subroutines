function pout = p_success_mc(n, m, alpha, xi, Nsim)

%This function returns the ordinal optimisation probability of success for
%the Gaussian case, using Monte-Carlo simulation

if (nargin < 5)
    Nsim = 20000; %number of simulations
end

%unconditional
count = 0;
for i = 1:Nsim
    
    X = randn(n, 1);
    Y = xi*randn(n, 1);
    
    Z = X + Y;
    
    [~, indsort] = sort(Z, 'ascend');
    
    if (any(X(indsort(1:m)) <= norminv(alpha)))
        count = count + 1;
    end
    
end

pout = count/Nsim;

end

