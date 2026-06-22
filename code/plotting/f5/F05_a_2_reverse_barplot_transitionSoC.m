% Plot C-rate diagnostic errors, including transition-SoC error, for Figure 5a.
%
% The workbook contains summary metrics assembled from the A04a/A04b C-rate
% robustness workflow. Rows correspond to C/20, C/10, C/5, and C/3 relative to
% the C/100 reference; columns are material eSoH metrics plus transition SoC.

clear
close all
clc

script_dir = fileparts(mfilename('fullpath'));
data_file = fullfile(script_dir, 'data', 'Z01_CrateAccuracy_TransitionSoC_v2.xlsx');
figure_dir = fullfile(script_dir, 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

parameter_names = {'C_{n,Si}', 'C_{n,Gr}', 'C_n', 'C_p', 'LI', 'Transition SoC', 'MAE'};
crate_labels = {'0.05C', '0.1C', '0.2C', '0.33C'};
param_colors = [
    0.2588 0.5059 0.6431
    0.6000 0.6000 0.6000
    0.7608 0.3882 0.2275
    0.4000 0.4000 0.4000
    0.2000 0.2000 0.2000
    0.4980 0.6941 0.8000
    0.8392 0.5922 0.4784
];

plot_error_bars(data_file, 'BOL_Rs', crate_labels, parameter_names, param_colors, ...
    fullfile(figure_dir, 'F5_a_2_BOL_crate_error.svg'));
plot_error_bars(data_file, 'EOL_Rs', crate_labels, parameter_names, param_colors, ...
    fullfile(figure_dir, 'F5_a_2_EOL_crate_error.svg'));

%% Helpers

function plot_error_bars(data_file, sheet_name, crate_labels, parameter_names, param_colors, output_file)
bar_width = 0.12;
opacity = 1.00;

data = readtable(data_file, 'Sheet', sheet_name, 'VariableNamingRule', 'preserve');
error_matrix = data{:, :};
[n_crates, n_params] = size(error_matrix);
crate_labels = crate_labels(1:n_crates);
x = 1:n_crates;

fig = figure('Position', [50, 50, 700, 320], 'Color', 'w');
hold on

for j = 1:n_params
    bar(x + (j - 1) * bar_width, error_matrix(:, j), ...
        bar_width, ...
        'FaceColor', param_colors(j, :), ...
        'EdgeColor', param_colors(j, :), ...
        'FaceAlpha', opacity, ...
        'DisplayName', parameter_names{j}, ...
        'LineWidth', 1.0);
end

xticks(x + (n_params - 1) * bar_width / 2);
xticklabels(crate_labels);
ylabel('Absolute Error [%]');
set(gca, 'YLim', [0, 20], ...
    'FontSize', 20, ...
    'XColor', [0.1 0.1 0.1], ...
    'YColor', [0.1 0.1 0.1]);
box on

legend(parameter_names, ...
    'Location', 'northwest', ...
    'FontSize', 16, ...
    'Box', 'on', ...
    'NumColumns', 2);

set(fig, 'PaperPositionMode', 'auto');
print(fig, output_file, '-dsvg', '-vector');
end
