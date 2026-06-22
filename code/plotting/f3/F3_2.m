% Plot the room-temperature RPT voltage and dV/dQ progression for Figure 3.
%
% The input files are exported processed RPT traces for cell064. Each file
% corresponds to one C/20 reference performance test during aging.

clear
close all
clc

script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, 'data', 'Room');
figure_dir = fullfile(script_dir, 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

% List all matching files.
files = dir(fullfile(data_dir, 'Room_Cell64_RPT*.csv'));
nFiles = length(files);

% Define start and end color (hex to RGB [0-1])
startColor = sscanf('525660', '%2x%2x%2x', [1 3]) / 255;
endColor   = sscanf('b6bcca', '%2x%2x%2x', [1 3]) / 255;

% Interpolate colormap
cmap = zeros(nFiles, 3);
for i = 1:3
    cmap(:, i) = linspace(startColor(i), endColor(i), nFiles);
end

% Create figure
figure('Position', [50, 50,  420, 230]); hold on; 
% Loop over each file and plot
for k = 1:nFiles
    filepath = fullfile(data_dir, files(k).name);
    T = readtable(filepath);

    if all(ismember({'Q', 'V'}, T.Properties.VariableNames))
        plot(T.Q, T.V, 'LineWidth', 1, 'Color', cmap(k, :));
    else
        warning('Missing expected columns in %s', files(k).name);
    end
end

xlabel('Q [Ah]');
ylabel('Voltage [V]');

box on;
grid off
ax = gca;
ax.FontSize = 20;
set(gcf,'color','w');

print(gcf, fullfile(figure_dir, 'F3_2_1.svg'), '-dsvg', '-painters');

% Create figure
figure('Position', [50, 50, 420, 230]); hold on; 
% Loop over each file and plot
for k = 1:nFiles
    filepath = fullfile(data_dir, files(k).name);
    T = readtable(filepath);

    if all(ismember({'Q', 'dVdQ'}, T.Properties.VariableNames))
        plot(T.Q, T.dVdQ, 'LineWidth', 1, 'Color', cmap(k, :));
    else
        warning('Missing expected columns in %s', files(k).name);
    end
end

xlabel('Q [Ah]');
ylabel('dV/dQ [V/Ah]');
ylim([0,1])
box on;
grid off
ax = gca;
ax.FontSize = 20;
set(gcf,'color','w');

print(gcf, fullfile(figure_dir, 'F3_2_2.svg'), '-dsvg', '-painters');
