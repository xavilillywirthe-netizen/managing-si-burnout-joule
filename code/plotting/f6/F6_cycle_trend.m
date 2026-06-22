% Plot cycling-condition lifetime trends for Figure 6.
%
% The master table contains the lifetime diagnostic outputs with experimental
% group labels. This script shows individual-cell trajectories together with
% binned group-average trends for capacity, LAM-Si, LLI, and transition SoC.

clear
close all
clc

script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, 'F6_group_plots_outputs_html');
figure_dir = fullfile(script_dir, 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

master_file = fullfile(data_dir, 'Master_with_groups_and_day.csv');
Master = readtable(master_file);
Master.transition_soc = Master.transition_soc*100;
% Metrics shown in the Figure 6 lifetime-trend panels.
param_names_main = {'C','CnSi','LLI','transition_soc'};
ylabel_strings = {'Capacity [Ah]', ...
    'C_{n,Si} [Ah]', ...
    'Lithium Inventory [Ah]',...
    'Transition SoC [%]',...
    };

% Accept either the compact plotting names or the original diagnostic-output
% names so the script can be run directly from the exported master table.
if ~ismember('C', Master.Properties.VariableNames) && ismember('Qcc_max_meas', Master.Properties.VariableNames)
    Master.C = Master.Qcc_max_meas;
end

if ~ismember('CnSi', Master.Properties.VariableNames) && ismember('Cn_Si', Master.Properties.VariableNames)
    Master.CnSi = Master.Cn_Si;
end

if ~ismember('CnGr', Master.Properties.VariableNames) && ismember('Cn_Gr', Master.Properties.VariableNames)
    Master.CnGr = Master.Cn_Gr;
end

if ~ismember('AhTh', Master.Properties.VariableNames) && ismember('Ah_throughput', Master.Properties.VariableNames)
    Master.AhTh = Master.Ah_throughput;
end

if ~ismember('EFC', Master.Properties.VariableNames)
    Master.EFC = Master.AhTh / 5;
end

% Exclude rows that were not assigned to one of the plotted cycling groups.
if ismember('cycle_group', Master.Properties.VariableNames)
    Master = Master(~strcmp(string(Master.cycle_group), "unassigned"), :);
end

% Cycling-temperature comparison used in the main Figure 6 panel.
compare_group_names = {
    'Pressure_25psi_25C'
    'P25_T45'
    'P25_T0'
    };

compare_group_colors = [
    0.2588 0.5059 0.6431   % blue
    0.7647 0.3922 0.2235   % orange
    0.3 0.3 0.3                   % gray
    ];

% Alternative DoD-window comparison. Uncomment this block when regenerating
% the corresponding supplementary grouping.
% compare_group_names = {
%     'Pressure_25psi_25C'
%     'DOD_5_96'
%     'DOD_20_80'
%     'DOD_50_100'
%     'DOD_0_50'
%     };
% 
% compare_group_colors = [
%     0.2588 0.5059 0.6431
%     0.5961 0.7451 0.8000
%     0.4392 0.6196 0.6980
%     0.6 0.6 0.6
%     0.9294    0.6941    0.1255
%     0.5725    0.2980    0.6314
%     ];


x_col = 'EFC';
group_col = 'cycle_group';
cell_col = 'cell';

% Axis limits used to keep panels comparable across groups.
ylims = {
    [], ...   % C
    [0,1.6],...% CnSi
    [1,2.8], ...   % LLI
    [0,70], ...   % transition SoC
    };

% Draw individual-cell trajectories and the binned group mean for each metric.
fig = figure('Position', [50, 50, 1000, 210]);
tiledlayout(1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:numel(param_names_main)
    pname = param_names_main{i};
    nexttile(i); hold on; box on;

    for g = 1:numel(compare_group_names)
        gname = compare_group_names{g};
        base_color = compare_group_colors(g, :);

        DataG = Master(strcmp(string(Master.(group_col)), gname), :);
        if isempty(DataG) || ~ismember(pname, DataG.Properties.VariableNames)
            continue;
        end

        % Keep only rows with valid x/y values for the current metric.
        valid = ~isnan(DataG.(x_col)) & ~isnan(DataG.(pname));
        DataG = DataG(valid, :);
        if isempty(DataG)
            continue;
        end

        % Plot each cell as a thin, light trajectory.
        cell_ids = unique(DataG.(cell_col));

        for c = 1:numel(cell_ids)
            cid = cell_ids(c);
            idx = DataG.(cell_col) == cid;
            Tcell = sortrows(DataG(idx, :), x_col);

            if height(Tcell) < 2
                continue;
            end

            plot(Tcell.(x_col), Tcell.(pname), '-', ...
                'Color', lighten_color(base_color, 0.72), ...
                'LineWidth', 0.5,'Marker','o','MarkerSize',4,'MarkerFaceColor', lighten_color(base_color, 0.72),'MarkerEdgeColor', lighten_color(base_color, 0.72));
        end

        % Overlay the binned group average as a thicker trend line.
        [x_mean, y_mean] = compute_group_average_curve(DataG, x_col, pname, 45);

        plot(x_mean, y_mean, '-', ...
            'Color', base_color, ...
            'LineWidth', 0.8);

        plot(x_mean, y_mean, 'o', ...
            'MarkerSize', 8, ...
            'MarkerFaceColor', base_color, ...
            'MarkerEdgeColor', base_color, ...
            'LineStyle', 'none');
    end

    ylabel(ylabel_strings{i}, 'FontSize', 18);
    xlabel('EFC [-]', 'FontSize', 18);
    set(gca, 'FontSize', 18);

    if ~isempty(ylims{i})
        ylim(ylims{i});
    end
end

% lgd = legend(compare_group_names, 'Location', 'southoutside', 'Orientation', 'horizontal');
% lgd.Layout.Tile = 'south';

set(gcf, 'Color', 'w');
set(fig, 'PaperPositionMode', 'auto');
print(fig, fullfile(figure_dir, 'F6_cycle_trend.svg'), '-dsvg', '-vector');

%% =========================================================
% Helper functions
% =========================================================
function [x_mean, y_mean] = compute_group_average_curve(T, x_col, y_col, bin_width)
x = T.(x_col);
y = T.(y_col);

xmin = min(x, [], 'omitnan');
xmax = max(x, [], 'omitnan');

edges = xmin:bin_width:(xmax + bin_width);
if numel(edges) < 2
    x_mean = x;
    y_mean = y;
    return;
end

x_mean = [];
y_mean = [];

for i = 1:(numel(edges)-1)
    idx = x >= edges(i) & x < edges(i+1);
    if ~any(idx)
        continue;
    end
    x_mean(end+1,1) = mean(x(idx), 'omitnan');
    y_mean(end+1,1) = mean(y(idx), 'omitnan');
end
end

function c_out = lighten_color(c_in, amount)
% amount in [0,1], larger values move the color closer to white.
c_out = c_in + (1 - c_in) * amount;
c_out = min(max(c_out, 0), 1);
end
