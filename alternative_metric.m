function [distance] = alternative_metric(a,b,beta)
dis=norm(a-b)^2;
distance=1-exp(-beta*dis);
end

