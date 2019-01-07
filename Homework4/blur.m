function y = blur(x)

h = exp(-[-2:2].^2/2); % filter
L = length(h); % 5
N = length(x);
y = zeros(N,1);
for j = 1:N
    for i = 1:L
        if j+i-3 <= 0
            y(j) = y(j)+x(j+i-3+N)*h(i);
        elseif j+i-3 >= N+1
            y(j) = y(j)+x(j+i-3-N)*h(i);
        else
            y(j) = y(j)+x(j+i-3)*h(i);
        end
    end
end

end