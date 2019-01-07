function y = Bstar(x)

h = exp(-[-2:2].^2/2); % filter
N = length(x);
y = ifft(fft(x).*conj(fft(h,N)));

end