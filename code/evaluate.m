[data1, samp_freq1] = audioread('vocals.wav');
[data2, samp_freq2] = audioread('bass.wav');
[data3, samp_freq3] = audioread('drums.wav');
[data4, samp_freq4] = audioread('other.wav');

% take only num_points to compute power (ratio used)
num_points = 100000;
points_taken = floor(linspace(1, size(data1,1), num_points));

%%%%%%% Replace this with your model o/p %%%%%%%%%%%%%
snr = 0.5;
data1_predicted = awgn(data1, snr, 'measured');
data2_predicted = awgn(data2, snr, 'measured');
data3_predicted = awgn(data3, snr, 'measured');
data4_predicted = awgn(data4, snr, 'measured');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


se = [data1_predicted(points_taken,1), data2_predicted(points_taken,1), data3_predicted(points_taken,1), data4_predicted(points_taken,1)]';
s = [data1(points_taken,1), data2(points_taken,1), data3(points_taken,1), data4(points_taken,1)]';

[SDR,SIR,SAR,perm]=bss_eval_sources(se,s);
