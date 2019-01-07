function [h,test1,test2] = gradCheck2(f,gradf,x,dx)

h = 1.5*10.^-[0:7]';
n = length(h);
test1 = zeros(n,1);
test2 = zeros(n,1);
for k = 1:n
    test1(k) = abs(f(x)+gradf(x)'*h(k)*dx-f(x+h(k)*dx));
    test2(k) = -2*(f(x+h(k)*dx)-f(x))+(gradf(x+h(k)*dx)+gradf(x))'*h(k)*dx;
end

end