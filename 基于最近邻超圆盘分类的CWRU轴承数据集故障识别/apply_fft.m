function [f, freq_values] = apply_fft(x, fs, num_samples)
    f = linspace(0.0, (fs/2.0), num_samples/2);
    freq_values = fft(x);
    freq_values = 2.0/num_samples * abs(freq_values(1:num_samples/2));
end
