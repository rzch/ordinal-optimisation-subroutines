function pout = p_success_approx(n, m, alpha, xi)

%This function returns the ordinal optimisation probability of success for
%the Gaussian case, using the approximation formula

[a, A] = joint_asymp_order_stat(n, m, xi);
B = xi^2/(1 + xi^2)*eye(m);
C = 1/(1 + xi^2)*eye(m);

[means, covmat] = marginalise_gaussian(a, A, C, B);

upper = norminv(1 - alpha)*ones(m, 1);

pout = 1 - mvncdf(upper, means, covmat);

    function [means, covmat] = joint_asymp_order_stat(n, m, xi)
        
        p = (n - (1:m))/n;
        p = fliplr(p)';
        
        means = norminv(p, 0, sqrt(1 + xi^2));
        
        covmat = zeros(m, m);
        
        for i = 1:m
            for j = 1:m
                if (i <= j)
                    covmat(i, j) = p(i)*(1 - p(j))/(n*normpdf(means(i), 0, sqrt(1 + xi^2))*normpdf(means(j), 0, sqrt(1 + xi^2)));
                    covmat(j, i) = covmat(i, j);
                end
            end
        end
        
    end

    function [means, covmat] = marginalise_gaussian(a, A, C, B)
        means = C'*a;
        covmat = B + C'*A*C;
    end

end

