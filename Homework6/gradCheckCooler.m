function [h test1 test2] = gradCheckCooler(f,gradf,x,dx,epsilon)

h = 1.5*10.^[

% Second order check
h = 1;
num = abs(f(x)+gradf(x)'*h*dx-f(x+h*dx));
for n = 1:10
    h = 2^(-n);
    denom = num;
    num = abs(f(x)+gradf(x)'*h*dx-f(x+h*dx));
    if num == 0
        break
    else
    end
    ratio = num/denom;
    if abs(ratio-4) >= epsilon
        logical = false;
    else
    end
end

% Third order check
h = 1;
num = -2*(f(x+h*dx)-f(x))+(gradf(x+h*dx)+gradf(x))'*h*(x+h*dx);
for n = 1:10
    h = 2^(-n);
    denom = num;
    num = -2*(f(x+h*dx)-f(x))+(gradf(x+h*dx)+gradf(x))'*h*dx;
    if num == 0
        break
    else
    end
    ratio = num/denom;
    if abs(ratio-8) >= epsilon
        logical = false;
    else
    end
end

end