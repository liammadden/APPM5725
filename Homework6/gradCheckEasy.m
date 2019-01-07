function [h,test1,test2] = gradCheckEasy(f,gradf,x)

n = length(x);
h = 1.5*10.^[2:-1:-2]';
niter = length(h);
test1 = zeros(niter,1);
test2 = zeros(niter,1);
for k = 1:niter
    for i = 1:n
        e = zeros(n,1);
        e(i) = 1;
        test1(k) = max([test1(k) abs((f(x+h(k)*e)-f(x))/h(k)-e'*gradf(x))]);
        test2(k) = max([test2(k) abs((f(x+h(k)*e)-f(x-h(k)*e))/2/h(k)-e'*gradf(x))]);
    end
end

end