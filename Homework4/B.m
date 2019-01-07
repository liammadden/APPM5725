function y = B(x)

h = exp(-[-2:2].^2/2); % filter
N = length(x);
y = ifft(fft(x).*fft(h,N));

end