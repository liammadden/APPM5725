%% Direct Method for Bee Colonies

%% Abstract
% We use the direct method to solve for the optimal parameters of
% waggle dance intensity and headbutting inensity for a bee colony
% navigating two food sources of time-varying quality. While the
% optimization problem was nonconvex, the method did converge to a local
% max, which was very close to the global max, found via grid search and
% Nelder-Mead in another file. The adjoint state method, though, is better.

%% Clear
clear all
close all

%% Bees
tmax = 100;
tau = 10;
alpha_a = @(t) 6*exp(t/tmax)-5;
alpha_b = @(t) 17-6*exp(t/tmax);
c_a = 0.5;
c_b = 0.5;
h = 1/36;
tspan = (tau:h:tmax);
x0 = [0;0];

niter = 2*1e2;
betas = zeros(niter,1);
betas(1) = 2;
gammas = zeros(niter,1);
gammas(1) = 2;
rhos = zeros(niter,1);
rhos(1) = 2;
obj = zeros(niter,1);
time = zeros(niter,1);
for i = 1:niter
    tic
    beta = betas(i);
    gamma = gammas(i);
    rho = rhos(i);
    F = @(t,x) [alpha_a(t)*(1-x(1) - x(2))+ beta*x(1)*(1-x(1)-x(2))-gamma*x(1) - rho*(t-tau)*x(1)*x(2);
        alpha_b(t)*(1-x(1) - x(2))+ beta*x(2)*(1-x(1)-x(2))-gamma*x(2) - rho*(t-tau)*x(1)*x(2)];
    [t,x] = ode45(F,tspan,x0);
    yield = (alpha_a(tspan)' - c_a).*x(:,1) +  (alpha_b(tspan)' - c_b).*x(:,2);
    F1 = @(t,y) [-alpha_a(t)*(y(1)+y(2))+x(round((t-tau)/h+1),1)+beta*y(1)-x(round((t-tau)/h+1),1)^2-2*beta*x(round((t-tau)/h+1),1)*y(1)-x(round((t-tau)/h+1),1)*x(round((t-tau)/h+1),2)-beta*y(1)*x(round((t-tau)/h+1),2)-beta*x(round((t-tau)/h+1),1)*y(2)-gamma*y(1)-rho*(t-tau)*(x(round((t-tau)/h+1),1)*y(2)+y(1)*x(round((t-tau)/h+1),2));
        -alpha_b(t)*(y(1)+y(2))+x(round((t-tau)/h+1),2)+beta*y(2)-x(round((t-tau)/h+1),2)^2-2*beta*x(round((t-tau)/h+1),2)*y(2)-x(round((t-tau)/h+1),1)*x(round((t-tau)/h+1),2)-beta*y(1)*x(round((t-tau)/h+1),2)-beta*x(round((t-tau)/h+1),1)*y(2)-gamma*y(2)-rho*(t-tau)*(x(round((t-tau)/h+1),1)*y(2)+y(1)*x(round((t-tau)/h+1),2))];
    [~,y1] = ode45(F1,tspan,[0;0]);
    yielddbeta = (alpha_a(tspan)' - c_a).*y1(:,1) +  (alpha_b(tspan)' - c_b).*y1(:,2);
    dgradbeta = max(0,yield).*yielddbeta;
    gradbeta = trapz(t,dgradbeta);
    F2 = @(t,y) [-alpha_a(t)*(y(1)+y(2))+beta*(y(1)-2*x(round((t-tau)/h+1),1)*y(1)-x(round((t-tau)/h+1),1)*y(2)-y(1)*x(round((t-tau)/h+1),2))-x(round((t-tau)/h+1),1)-gamma*y(1)-rho*(t-tau)*(x(round((t-tau)/h+1),1)*y(2)+y(1)*x(round((t-tau)/h+1),2));
        -alpha_b(t)*(y(1)+y(2))+beta*(y(2)-2*x(round((t-tau)/h+1),2)*y(2)-x(round((t-tau)/h+1),1)*y(2)-y(1)*x(round((t-tau)/h+1),2))-x(round((t-tau)/h+1),2)-gamma*y(2)-rho*(t-tau)*(x(round((t-tau)/h+1),1)*y(2)+y(1)*x(round((t-tau)/h+1),2))];
    [~,y2] = ode45(F2,tspan,[0;0]);
    yielddgamma = (alpha_a(tspan)' - c_a).*y2(:,1) +  (alpha_b(tspan)' - c_b).*y2(:,2);
    dgradgamma = max(0,yield).*yielddgamma;
    gradgamma = trapz(t,dgradgamma);
    F3 = @(t,y) [-alpha_a(t)*(y(1)+y(2))+beta*(y(1)-2*x(round((t-tau)/h+1),1)*y(1)-x(round((t-tau)/h+1),1)*y(2)-y(1)*x(round((t-tau)/h+1),2))-gamma*y(1)-x(round((t-tau)/h+1),1)*x(round((t-tau)/h+1),2)-rho*(t-tau)*(x(round((t-tau)/h+1),1)*y(2)+y(1)*x(round((t-tau)/h+1),2));
        -alpha_b(t)*(y(1)+y(2))+beta*(y(2)-2*x(round((t-tau)/h+1),2)*y(2)-x(round((t-tau)/h+1),1)*y(2)-y(1)*x(round((t-tau)/h+1),2))--gamma*y(2)-x(round((t-tau)/h+1),1)*x(round((t-tau)/h+1),2)-rho*(t-tau)*(x(round((t-tau)/h+1),1)*y(2)+y(1)*x(round((t-tau)/h+1),2))];
    [~,y3] = ode45(F3,tspan,[0;0]);
    yielddrho = (alpha_a(tspan)' - c_a).*y3(:,1) +  (alpha_b(tspan)' - c_b).*y3(:,2);
    dgradrho = max(0,yield).*yielddrho;
    gradrho = trapz(t,dgradrho);
    step = 1/max(abs(gradbeta),max(abs(gradgamma),abs(gradrho)))/10/sqrt(i); % try line search
    betas(i+1) = max(0,beta+step*gradbeta);
    gammas(i+1) = max(0,gamma+step*gradgamma);
    rhos(i+1) = max(0,rho+step*gradrho);
    obj(i) = .5*trapz(t,max(0,yield).^2);
    time(i) = toc;
end
    
%% Plot
figure(1)
plot(1:niter,obj,'.k')

figure(2)
plot(t,yield,'-k',t,alpha_a(t),'-r',t,alpha_b(t),'-b')

figure(3)
plot(t,x(:,1),'-r',t,x(:,2),'-b')