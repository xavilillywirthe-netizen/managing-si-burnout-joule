% Plot representative full-cell and electrode-potential reconstruction errors for Figure 4.
%
% The input CSV contains a validation pulse/report with measured and
% reconstructed full-cell voltage, anode potential, and cathode potential.
% Each panel overlays measurement and reconstruction on the left axis and
% shows the pointwise reconstruction error on the right axis.

clear
close all
clc

script_dir = fileparts(mfilename('fullpath'));
data_file = fullfile(script_dir, 'data', 'voltage_fit_example', 'rpt_02_electrode_potential_comparison.csv');
figure_dir = fullfile(script_dir, 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

data = readtable(data_file);

capacity_ah = data.Q_Ah;
voltage_fit = data.V_full_recon;
voltage_meas = data.V_full_meas;
anode_fit = data.V_anode_recon;
anode_meas = data.V_anode_meas;
cathode_fit = data.V_cathode_recon;
cathode_meas = data.V_cathode_meas;

voltage_error = voltage_meas - voltage_fit;
anode_error = anode_fit - anode_meas;
cathode_error = cathode_fit - cathode_meas;

fprintf('Full-cell voltage RMSE: %.3f mV\n', rmse_mv(voltage_error));
fprintf('Anode potential RMSE: %.3f mV\n', rmse_mv(anode_error));
fprintf('Cathode potential RMSE: %.3f mV\n', rmse_mv(cathode_error));

plot_reconstruction_panel( ...
    capacity_ah, voltage_meas, voltage_fit, voltage_error, ...
    [0.3176 0.3765 0.4039], ...
    'Voltage [V]', 'Voltage Error [mV]', ...
    fullfile(figure_dir, 'F4_2_1.svg'));

plot_reconstruction_panel( ...
    capacity_ah, anode_meas, anode_fit, anode_error, ...
    [0.7608 0.3882 0.2275], ...
    'Potential [V]', 'Potential Error [mV]', ...
    fullfile(figure_dir, 'F4_2_2.svg'));

plot_reconstruction_panel( ...
    capacity_ah, cathode_meas, cathode_fit, cathode_error, ...
    [0.2588 0.5059 0.6431], ...
    'Potential [V]', 'Potential Error [mV]', ...
    fullfile(figure_dir, 'F4_2_3.svg'));

%% Helpers

function plot_reconstruction_panel(capacity_ah, measured, fitted, error_v, color, left_label, right_label, output_file)
fig = figure('Position', [50, 50, 400, 300], 'Color', 'w');
hold on

yyaxis left
plot(capacity_ah, measured, ...
    'LineWidth', 0.5, ...
    'Color', color);
plot(capacity_ah, fitted, ...
    'LineWidth', 0.5, ...
    'Color', color, ...
    'LineStyle', ':');

ax = gca;
ax.FontSize = 20;
ax.XLabel.String = 'Q [Ah]';
ax.YLabel.String = left_label;
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];

yyaxis right
fill([capacity_ah; flipud(capacity_ah)], ...
    [zeros(size(error_v)); flipud(error_v * 1000)], ...
    color, ...
    'FaceAlpha', 0.3, ...
    'EdgeColor', 'none');

ax = gca;
ax.XLim = [0, max(capacity_ah)];
ax.FontSize = 20;
ax.YLabel.String = right_label;
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];
box on

set(fig, 'PaperPositionMode', 'auto');
print(fig, output_file, '-dsvg', '-vector');
end

function value = rmse_mv(error_v)
value = sqrt(mean(error_v .^ 2, 'omitnan')) * 1000;
end
