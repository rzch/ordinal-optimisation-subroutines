function pout = p_success_numerical(n, m, alpha, xi)

%This function returns the ordinal optimisation probability of success for
%the Gaussian case, using numerical integration

pout = 1 - (1 - alpha)^n;

for g = 1:(n - m)
    
    integrand = @(x) Zcdf_upper(x, n, m, g, alpha, xi).*Zpdf_lower(x, n, m, g, alpha, xi);
    
    %update the sum
    pout = pout - binopdf(g, n, alpha)*integral(@(x) integrand(x), -Inf, Inf);
    
end

    function cprobs = Zcdf_upper(x, n, m, g, alpha, xi)
        %This function returns the cumulative probability of Z_{g + m} at the supplied points
        
        cprobs = zeros(size(x));
        for i = 1:length(x)
            cprobs(i) = integral(@(z) Zpdf_upper(z, n, m, g, alpha, xi), -Inf, x(i));
        end
        
    end

    function densities = Zpdf_lower(x, n, m, g, alpha, xi)
        %This function returns the density of Z_{1} at the supplied points
        
        %O(1) computation
        F = sumrvs_cdf(x, alpha, xi);
        f = sumrvs_pdf(x, alpha, xi);
        densities = exp(log(g) + (g - 1)*log(1 - F) + log(f));
        
        
        function cprobs = sumrvs_cdf(x, alpha, xi)
            %This function computes the cumulative probabilities of the sum distribution of a
            %scaled Gaussian (with std. dev. xi) with the right-truncated
            %standard Gaussian at the alpha*100 percentile
            
            cprobs = arrayfun(@(y) integral(@(z) normcdf(y - z, 0, xi).*righttrunc_normpdf(z, norminv(alpha)), -Inf, norminv(alpha)), x);
            
        end
        
        function densities = sumrvs_pdf(x, alpha, xi)
            %This function computes the densities of the sum distribution of a
            %scaled Gaussian (with std. dev. xi) with the right-truncated
            %standard Gaussian at the alpha*100 percentile
            
            densities = arrayfun(@(y) integral(@(z) normpdf(y - z, 0, xi).*righttrunc_normpdf(z, norminv(alpha)), -Inf, norminv(alpha)), x);
            
        end
        
        function densities = righttrunc_normpdf(x, thres)
            %This function computes the densities of the right-truncated
            %standard Gaussian with the supplied threshold at the given points
            
            densities = (x <= thres).*normpdf(x)/normcdf(thres);
            
        end
        
    end

    function densities = Zpdf_upper(x, n, m, g, alpha, xi)
        %This function returns the density of Z_{g + m} at the supplied points
        
        %O(1) computation
        F = sumrvs_cdf(x, alpha, xi);
        f = sumrvs_pdf(x, alpha, xi);
        densities = exp(log(m) + gammaln(n - g + 1) - gammaln(m + 1) - gammaln(n - g - m + 1) ...
            + (m - 1)*log(F) + (n - g - m)*log(1 - F) + log(f));
        
        
        function cprobs = sumrvs_cdf(x, alpha, xi)
            %This function computes the cumulative probabilities of the sum distribution of a
            %scaled Gaussian (with std. dev. xi) with the left-truncated
            %standard Gaussian at the alpha*100 percentile
            
            cprobs = arrayfun(@(y) integral(@(z) normcdf(y - z, 0, xi).*lefttrunc_normpdf(z, norminv(alpha)), norminv(alpha), Inf), x);
            
        end
        
        function cprobs = sumrvs_pdf(x, alpha, xi)
            %This function computes the densities of the sum distribution of a
            %scaled Gaussian (with std. dev. xi) with the left-truncated
            %standard Gaussian at the alpha*100 percentile
            
            cprobs = arrayfun(@(y) integral(@(z) normpdf(y - z, 0, xi).*lefttrunc_normpdf(z, norminv(alpha)), norminv(alpha), Inf), x);
            
        end
        
        function densities = lefttrunc_normpdf(x, thres)
            %This function computes the densities of the left-truncated
            %standard Gaussian with the supplied threshold at the given points
            
            densities = (x >= thres).*normpdf(x)/(1 - normcdf(thres));
            
        end
        
    end

end