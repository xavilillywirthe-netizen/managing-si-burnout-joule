% Plot material-resolved incremental capacity and silicon current share for Figure 1b.
%
% The BOL eSoH reconstruction projects the silicon and graphite material
% responses onto the full-cell SoC axis. The plotted ratio quantifies the
% fraction of cell current carried by silicon at each SoC.

clear
close all

script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, 'data');
figure_dir = fullfile(script_dir, 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

data = readtable(fullfile(data_dir, 'BOL_eSOH_details_70EOL.csv'));

soc_percent = data.SoCs * 100;
dqdv_cell = -data.dqdv_fit;
dqdv_si_signed = -data.dqdv_fit_si;
dqdv_si_plot = -dqdv_si_signed;
dqdv_gr = dqdv_cell + dqdv_si_signed;
si_current_share = dqdv_si_plot ./ dqdv_cell * 100;

fig = figure('Position', [50, 50, 380, 280], 'Color', 'w');
hold on

% Left axis: full-cell and material-resolved incremental-capacity traces.
plot(soc_percent, dqdv_cell, ...
    'Color', [0.8 0.8 0.8], ...
    'LineWidth', 0.5, ...
    'LineStyle', '-');

plot(soc_percent, dqdv_si_plot, ...
    'Color', [0.4980 0.6941 0.8000], ...
    'LineWidth', 0.5, ...
    'LineStyle', '-');

valid_si = isfinite(soc_percent) & isfinite(dqdv_si_plot);
fill([soc_percent(valid_si); flipud(soc_percent(valid_si))], ...
    [zeros(nnz(valid_si), 1); flipud(dqdv_si_plot(valid_si))], ...
    [0.2588 0.5059 0.6431], ...
    'FaceAlpha', 0.3, ...
    'EdgeColor', 'none');

plot(soc_percent, dqdv_gr, ...
    'Color', [0.6510 0.6510 0.6510], ...
    'LineWidth', 0.5, ...
    'LineStyle', '-');

valid_gr = isfinite(soc_percent) & isfinite(dqdv_gr);
fill([soc_percent(valid_gr); flipud(soc_percent(valid_gr))], ...
    [zeros(nnz(valid_gr), 1); flipud(dqdv_gr(valid_gr))], ...
    [0.6510 0.6510 0.6510], ...
    'FaceAlpha', 0.3, ...
    'EdgeColor', 'none');

ax = gca;
ax.YLim = [0, 8];
ax.FontSize = 20;
ax.XLabel.String = 'SOC [%]';
ax.YLabel.String = 'dQ/dV [Ah/V]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];
box on

% Right axis: silicon current share, defined as the silicon contribution
% normalized by the full-cell incremental-capacity response.
yyaxis right
plot(soc_percent, si_current_share, ...
    'Color', [0.6353 0.3098 0.2745], ...
    'LineWidth', 1, ...
    'LineStyle', '-');

ax = gca;
ax.XLim = [0, 100];
ax.FontSize = 20;
ax.XLabel.String = 'SOC [%]';
ax.YLabel.String = 'Silicon current share [%]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.6353 0.3098 0.2745];
box on

set(fig, 'PaperPositionMode', 'auto');
print(fig, fullfile(figure_dir, 'F1_2.svg'), '-dsvg', '-vector');
