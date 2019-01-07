%% Liam Madden
% APPM 5720
% Homework 4

%% Clear
clear all
close all

%% Blur
N = 100;
x = zeros(N,1);
x(10) = 1;
x(13) = -1;
x(50) = 0.3;
x(70) = -0.2;
yNoNoise = blur(x);
B = implicit2explicit(@(x) blur(x),N); % yNoNoise=B*x
sigma = 0.02; % standard deviation
z = normrnd(0,sigma,N,1);
y = yNoNoise + z;

%% Deblur
epsilon = sigma*sqrt(N);
% recover x
cvx_begin
    variables xr(N)
    dual variable lambda
    minimize norm(xr,1)
    subject to
        lambda : sum_square(B*xr-y) <= epsilon^2;
cvx_end

%% Plot
count = 1:N;
figure(1)
plot(count,x,'-k',count,y,'-c',count,xr,'-g')
legend('Original signal','Blurred signal','Deblurred signal')

%% Dual
% check that the dual x is the same as the recovered x
cvx_begin
    variables xd(N)
    minimize norm(xd,1)+lambda*sum_square(B*xd-y)
cvx_end
disp(norm(xd-xr)); % muy poco

%% adjoint
a = @(x) blur(x); % functionhandle for blur
b = @(functionhandle,x) adjoint(functionhandle,x); % functionhandle for adjoint
c = @(x) b(a,x); % functionhandle for adjoint of blur
Bstar = implicit2explicit(c,N);
disp(norm(B-Bstar')); % they are equal by construction

%% cvx
tau = 1/2/lambda;
tic
cvx_begin
    variables x3(N)
    minimize tau*norm(x3,1)+0.5*sum_square(B*x3-y)
cvx_end
t1 = toc;
disp(norm(x3-xr)); % I mean, this was the exact same as how we calculated xd

%% fasta
opts = [];
opts.recordObjective = true;
opts.verbose=true;
opts.stringHeader='    ';
initialGuess = zeros(N,1);
tic
[x4,outs] = fasta_sparseLeastSquares(a,c,y,tau,initialGuess,opts);
t2 = toc;

%% compare cvx to fasta
disp(norm(x4-x3)); % 3 digits of accuracy
disp(t1);
disp(t2); % much faster!!!!!