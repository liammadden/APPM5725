function t = lineSearch(t0,f,gradf,x,p)

c = 10^(-4);
rho = 0.9;
t = t0;

while f(x+t*p) >= f(x) + c*t*gradf(x)'*p
    t = rho*t;
end

end