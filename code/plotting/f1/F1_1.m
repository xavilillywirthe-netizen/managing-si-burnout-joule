% Plot material OCP curves and differential lithiation response for Figure 1a.
%
% The panel compares the graphite OCP with the effective silicon OCP used in
% the MSMR-informed anode model. The upper x-axis overlays dx/dU to show which
% material contributes capacity at each anode potential.

clear
close all

script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, 'data');
figure_dir = fullfile(script_dir, 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

si_ocp = readtable(fullfile(data_dir, 'Baseline_Si_OCP_original_vs_reconstructed_deformed.csv'));
gr_nmc_ocp = readtable(fullfile(data_dir, 'Graphite_NMC_OCP_reference.csv'));

sto_si = si_ocp.sto_common;
p_si = si_ocp.p_si_reconstructed_deformed;
dudx_si = -si_ocp.dUdx_si_reconstructed_deformed;
dqdu_si = 1 ./ dudx_si;

sto_gr = gr_nmc_ocp.x_gr;
p_gr = gr_nmc_ocp.p_gr;
dqdu_gr = gr_nmc_ocp.dxdU_gr_1mV;
p_gr_dqdu = gr_nmc_ocp.p_gr_1mV;

valid_si_dqdu = isfinite(dqdu_si) & isfinite(p_si);
dqdu_si = dqdu_si(valid_si_dqdu);
p_si_dqdu = p_si(valid_si_dqdu);

valid_gr_dqdu = isfinite(dqdu_gr) & isfinite(p_gr_dqdu);
dqdu_gr = dqdu_gr(valid_gr_dqdu);
p_gr_dqdu = p_gr_dqdu(valid_gr_dqdu);

valid_si_ocp = isfinite(sto_si) & isfinite(p_si);
sto_si = sto_si(valid_si_ocp);
p_si = p_si(valid_si_ocp);

valid_gr_ocp = isfinite(sto_gr) & isfinite(p_gr);
sto_gr = sto_gr(valid_gr_ocp);
p_gr = p_gr(valid_gr_ocp);

fig = figure('Position', [50 50 420 300], 'Color', 'w');
axis_position = [0.18 0.20 0.68 0.58];

% Bottom axis: material OCP versus stoichiometry.
ax_ocp = axes(fig, 'Position', axis_position);
hold(ax_ocp, 'on')
plot(ax_ocp, sto_si, p_si, 'LineWidth', 0.5, 'Color', [0.2588 0.5059 0.6431]);
plot(ax_ocp, sto_gr, p_gr, 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5]);

ax_ocp.XLim = [0 1];
ax_ocp.YLim = [0 0.7];
ax_ocp.FontSize = 20;
xlabel(ax_ocp, 'Stoichiometry [-]');
ylabel(ax_ocp, 'Potential [V]');
ax_ocp.XColor = [0.1 0.1 0.1];
ax_ocp.YColor = [0.1 0.1 0.1];
box(ax_ocp, 'on');

% Top axis: differential stoichiometry response on the same potential axis.
ax_dqdu = axes(fig, ...
    'Position', axis_position, ...
    'Color', 'none', ...
    'XAxisLocation', 'top', ...
    'YAxisLocation', 'right', ...
    'XLim', [0 40], ...
    'YLim', [0 0.7], ...
    'FontSize', 20);

hold(ax_dqdu, 'on')
fill(ax_dqdu, [0; dqdu_si(:); 0], [p_si_dqdu(1); p_si_dqdu(:); p_si_dqdu(end)], ...
    [0.2588 0.5059 0.6431], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

fill(ax_dqdu, [0; -dqdu_gr(:); 0], [p_gr_dqdu(1); p_gr_dqdu(:); p_gr_dqdu(end)], ...
    [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

linkaxes([ax_ocp, ax_dqdu], 'y');
ax_dqdu.YTick = [];
ax_dqdu.YColor = 'none';
ax_dqdu.XColor = [0.1 0.1 0.1];
xlabel(ax_dqdu, 'dx/dU [1/V]');
box(ax_dqdu, 'on');

set(fig, 'PaperPositionMode', 'auto');
print(fig, fullfile(figure_dir, 'F1_1.svg'), '-dsvg', '-vector');
