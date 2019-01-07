function x = fista(x0,f,gradf,Lf,prox,tol,maxIter,minIter)

x(:,1) = x0;
t(1) = 0;
y(:,1) = x0;
seqCvg = tol + 1;
k = 1;
while seqCvg >= tol
    if k == maxIter
        break
    else
    end
    x(:,k+1) = prox(y(:,k)-gradf(y(:,k))/Lf,1/Lf);
    if k >= minIter
        seqCvg = norm(x(:,k+1)-x(:,k));
    else
    end
    t(k+1) = (1+sqrt(1+4*t(k)^2))/2;
    y(:,k+1) = x(:,k+1) + (t(k)-1)/(t(k+1))*(x(:,k+1)-x(:,k));
    k = k + 1;
end