[data1, samp_freq1] = audioread('vocals.wav');
[data2, samp_freq2] = audioread('bass.wav');
[data3, samp_freq3] = audioread('drums.wav');
[data4, samp_freq4] = audioread('other.wav');

snr = 1000;
data1_noisy = awgn(data1, snr, 'measured');
data2_noisy = awgn(data2, snr, 'measured');
data3_noisy = awgn(data3, snr, 'measured');
data4_noisy = awgn(data4, snr, 'measured');

se = [data1_noisy(1:500,1), data2_noisy(1:500,1), data3_noisy(1:500,1), data4_noisy(1:500,1)]';
s = [data1(1:500,1), data2(1:500,1), data3(1:500,1), data4(1:500,1)]';
[SDR,SIR,SAR,perm]=bss_eval_sources(se,s);