function x = accelNest(x0,f,gradf,tol,maxIter,L,lineSearchToggle)
% step size starts at 1/L and then is determined by lineSearch if the
% latter is turned on. If L is unknown (L=0), the step size starts at 1.
if L == 0
    t = 1;
else
    t = 1/L;
end
x(:,1) = x0;
y(:,1) = x0;
error = tol+1;
k = 1;
while error >= tol
    if k == maxIter
        break
    else
    end
    p = -gradf(x(:,k));
    if lineSearchToggle == 1
        t = lineSearch(2*t,f,gradf,x(:,k),p);
    else
    end
    x(:,k+1) = y(:,k) + t*p;
    if k >= 1000
        error = norm(x(:,k)-x(:,k+1));
    else
    end
    y(:,k+1) = x(:,k+1) + (k+1)/(k+4)*(x(:,k+1)-x(:,k));
    k = k+1;
end

end