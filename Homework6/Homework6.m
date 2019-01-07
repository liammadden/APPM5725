%% Liam Madden
% APPM 5720
% Homework 6

%% Clear
clear all
close all

%% Try on 1-D functions
a = @(x) x^3;
b = @(x) 3*x^2;
c = @(x) 6*x;
d = @(x) 6;
x = 10;
dx = 1;
[h1,atest1,atest2] = gradCheck(a,b,x,dx);
figure(1)
plot(log(h1),log(atest1),'-k',log(h1),log(atest2),'--k')
xlabel({'log(h)','The test1 slope is ~2 and the test2 slope is ~3.'})
ylabel('log(test)')
legend('test1','test2')
title('atest')
[~,btest1,btest2] = gradCheck(b,c,x,dx);
figure(2)
plot(log(h1),log(btest1),'-k',log(h1),log(btest2),'--k')
xlabel({'log(h)','The test1 slope is ~2, but test2 fails because h is too small for it.'})
ylabel('log(test)')
legend('test1','test2')
title('btest')
[~,ctest1,ctest2] = gradCheck(c,d,x,dx);
figure(3)
plot(log(h1),log(ctest1),'-k',log(h1),log(ctest2),'--k')
xlabel({'log(h)','Both tests fail because h is too small for the function.'})
ylabel('log(test)')
legend('test1','test2')
title('ctest')

%% Try on a high-dimensional quadratic
N = 100;
A = rand(N);
e = @(x) x'*A*x;
f = @(x) (A+A')*x;
X = ones(N,1);
dX = randi(2,[N,1]);
[h2,etest1,etest2] = gradCheck(e,f,X,dX);
figure(4)
plot(log(h2),log(etest1),'-k',log(h2),log(etest2),'--k')
xlabel({'log(h)','The test1 slope is ~2, but test2 fails because h is too small for it.'})
ylabel('log(test)')
legend('test1','test2')
title('etest')

%% Try scaling
sa = @(x) 109*a(x);
sb = @(x) 100*b(x);
[~,satest1,satest2] = gradCheck(sa,sb,x,dx);
figure(5)
plot(log(h1),log(satest1),'-k',log(h1),log(satest2),'--k')
xlabel({'log(h)','The test1 slope is ~2, the test2 slope is ~3.'})
ylabel('log(test)')
legend('test1','test2')
title('satest')

%% Can it distinguish slightly off gradients?
ib = @(x) .9*b(x);
[~,iatest1,iatest2] = gradCheck(a,ib,x,dx);
figure(6)
plot(log(h1),log(iatest1),'-k',log(h1),log(iatest2),'--k')
xlabel({'log(h)','The test1 slope is ~1, the test2 slope is ~1.3'})
ylabel('log(test)')
legend('test1','test2')
title('iatest')