% Plot material-level and cell-level eSOH diagnostics for Figure 3.
%
% The input CSV files contain reconstructed electrode OCP curves and
% material-resolved voltage/dVdQ traces for representative silicon-loss cases.
% All paths are resolved relative to this script so the figure can be
% regenerated from a downloaded copy of the repository.

clear
close all
clc

script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, 'data');
figure_dir = fullfile(script_dir, 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

file_name1 = fullfile(data_dir, 'Baseline_eSOH_details_70EOL.csv');
file_name2 = fullfile(data_dir, '02Si_eSOH_details_70EOL.csv');
file_name3 = fullfile(data_dir, '08Si_eSOH_details_70EOL.csv');
file_name4 = fullfile(data_dir, 'Graphite_NMC_OCP_reference.csv');
file_name5 = fullfile(data_dir, 'Baseline_Si_OCP_original_vs_reconstructed_deformed.csv');
file_name6 = fullfile(data_dir, '02Si_Si_OCP_original_vs_reconstructed_deformed.csv');
file_name7 = fullfile(data_dir, '08Si_Si_OCP_original_vs_reconstructed_deformed.csv');
file_name8 = fullfile(data_dir, 'SiOCP_Extreme_Si_OCP_original_vs_reconstructed_deformed.csv');


baseline_table = readtable(file_name1);
Si02_table = readtable(file_name2);
Si08_table = readtable(file_name3);
NMC_Gr_table = readtable(file_name4);
Anode_table_baseline = readtable(file_name5);
Anode_table_02Si = readtable(file_name6);
Anode_table_08Si = readtable(file_name7);

Anode_table_extreme_si =  readtable(file_name8);

sto_Si = Anode_table_baseline.sto_common;
p_Si = Anode_table_baseline.p_si_reconstructed_deformed;
dVdQ_Si = -Anode_table_baseline.dUdx_si_reconstructed_deformed;
dQdV_Si = 1./dVdQ_Si;

sto_Si_extre = Anode_table_extreme_si.sto_common;
p_Si_extre = Anode_table_extreme_si.p_si_reconstructed_deformed;
dVdQ_Si_extre = -Anode_table_extreme_si.dUdx_si_reconstructed_deformed;

sto_Gr = NMC_Gr_table.x_gr;
p_Gr = NMC_Gr_table.p_gr;
dVdQ_Gr = -NMC_Gr_table.dUdx_gr;

pGr_dQdV = NMC_Gr_table.p_gr_1mV;
dQdV_Gr = NMC_Gr_table.dxdU_gr_1mV;

sto_nmc = NMC_Gr_table.x_nmc;
p_nmc = NMC_Gr_table.p_nmc;
dVdQ_nmc = -NMC_Gr_table.dUdx_nmc;

Vf_anode = Anode_table_baseline.p_anode_reconstructed;
sto_Anode = Anode_table_baseline.x_anode_common;

Vf_anode2 = Anode_table_02Si.p_anode_reconstructed;
sto_Anode2 = Anode_table_02Si.x_anode_common;

Vf_anode8 = Anode_table_08Si.p_anode_reconstructed;
sto_Anode8 = Anode_table_08Si.x_anode_common;

Qdata = baseline_table.Q_exp;
Vfit = baseline_table.V_fit;
Vfit_Anode = baseline_table.V_fit_anode;
Vfit_Cathode = baseline_table.V_fit_cathode;
dVdQ_full = baseline_table.dvdq_fit;
dVdQ_Cathode = baseline_table.dvdq_fit_cathode;
dVdQ_Anode = baseline_table.dvdq_fit_anode;

Qdata2 = Si02_table.Q_exp;
Vfit2 = Si02_table.V_fit;
Vfit_Anode2 = Si02_table.V_fit_anode;
Vfit_Cathode2 = Si02_table.V_fit_cathode;
dVdQ_full2 = Si02_table.dvdq_fit;
dVdQ_Cathode2 = Si02_table.dvdq_fit_cathode;
dVdQ_Anode2 = Si02_table.dvdq_fit_anode;

Qdata8 = Si08_table.Q_exp;
Vfit8 = Si08_table.V_fit;
Vfit_Anode8 = Si08_table.V_fit_anode;
Vfit_Cathode8 = Si08_table.V_fit_cathode;
dVdQ_full8 = Si08_table.dvdq_fit;
dVdQ_Cathode8 = Si08_table.dvdq_fit_cathode;
dVdQ_Anode8 = Si08_table.dvdq_fit_anode;


dQdV_Si(isnan(dQdV_Si)) = [];
dQdV_Gr(isnan(dQdV_Gr)) = [];
pGr_dQdV(isnan(NMC_Gr_table.dxdU_gr_1mV)) = [];
sto_Si(isnan(sto_Si)) = [];
p_Si(isnan(p_Si)) = [];
sto_Gr(isnan(sto_Gr)) = [];
p_Gr(isnan(p_Gr)) = [];
Vf_anode(isnan(Vf_anode)) = [];
sto_Anode(isnan(sto_Anode)) = [];
sto_Anode2(isnan(sto_Anode2)) = [];
sto_Anode8(isnan(sto_Anode8)) = [];
sto_Anode2 = movmean(sto_Anode2,20);
sto_Anode8 = movmean(sto_Anode8,20);

sto_nmc(isnan(sto_nmc)) = [];
p_nmc(isnan(p_nmc)) = [];
dVdQ_nmc(isnan(dVdQ_nmc)) = [];
p_nmc(isnan(p_nmc)) = [];

%% Material Level
% Si
figure('Position', [50, 50, 350, 300]);
hold on
yyaxis left
plot(sto_Si, p_Si, 'LineWidth',1, 'Color', [0.2588    0.5059    0.6431],'LineStyle','-');
plot(sto_Si_extre, p_Si_extre, 'LineWidth',1, 'Color', [0.4745    0.6471    0.7412],'LineStyle','-');

ax = gca;
ax.XLim = [0,1];
ax.YLim = [0,0.7];
ax.FontSize = 20;
ax.XLabel.String = 'Stoichiometry [-]';
ax.YLabel.String = 'OCP [V]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];
% ax.XGrid = 'on';
% ax.YGrid = 'on';
set(gcf,'color','w');

yyaxis right
plot(sto_Si, dVdQ_Si, 'LineWidth',0.5, 'Color', [0.2588    0.5059    0.6431],'LineStyle','--');
% plot(sto_Si_extre, dVdQ_Si_extre, 'LineWidth',0.5, 'Color', [0.4745    0.6471    0.7412],'LineStyle','--');
valid_idx = ~isnan(dVdQ_Si) & sto_Si >= 0 & sto_Si <= 1 & dVdQ_Si >= 0;
x_fill = sto_Si(valid_idx);
y_fill = dVdQ_Si(valid_idx);

y_fill(y_fill>1.5) = 1.5;
% Construct closed shape for fill: [x ->, x <-], [0s ->, y <-]
fill([x_fill; flipud(x_fill)], ...
     [zeros(size(x_fill)); flipud(y_fill)], ...
     [0.2588    0.5059    0.6431], ...
     'FaceAlpha', 0.1, 'EdgeColor', 'none');
ax = gca;
ax.YLabel.String = 'dU/dx [V]';
ax.YLim = [0,1.5];
ax.YColor = [0.1 0.1 0.1];

l = legend("Pristine OCP","Aged OCP","dUdx");
% l.FontAngle = 'italic';
% l.Color = [0.95 0.95 0.95];
l.FontSize = 18;
l.EdgeColor = [1 1 1];
l.NumColumns = 1;
box on
print(gcf, fullfile(figure_dir, 'F3_1.svg'), '-dsvg', '-vector');

% gr
figure('Position', [50, 50, 350, 300]);
hold on

yyaxis left
plot(sto_Gr, p_Gr, 'LineWidth',1, 'Color', [0.5 0.5 0.5]);
ax = gca;
ax.XLim = [0,1];
ax.YLim = [0,0.7];
ax.FontSize = 20;
ax.XLabel.String = 'Stoichiometry [-]';
ax.YLabel.String = 'OCP [V]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];
% ax.XGrid = 'on';
% ax.YGrid = 'on';

yyaxis right
plot(sto_Gr, dVdQ_Gr, 'LineWidth',0.5, 'Color', [0.5 0.5 0.5],'LineStyle','--');

valid_idx = ~isnan(dVdQ_Gr) & sto_Gr >= 0 & sto_Gr <= 1 & dVdQ_Gr >= 0;
x_fill = sto_Gr(valid_idx);
y_fill = dVdQ_Gr(valid_idx);

y_fill(y_fill>1.5) = 1.5;
% Construct closed shape for fill: [x ->, x <-], [0s ->, y <-]
fill([x_fill; flipud(x_fill)], ...
     [zeros(size(x_fill)); flipud(y_fill)], ...
     [0.5 0.5 0.5], ...
     'FaceAlpha', 0.1, 'EdgeColor', 'none');
ax.YLabel.String = 'dU/dx [V]';
ax.YLim = [0,1.5];
ax.YColor = [0.1 0.1 0.1];

% t = title('Composite Anode Open Circuit Potential');
% t.FontSize = 24;

l = legend("OCP","dUdx");
% l.FontAngle = 'italic';
% l.Color = [0.95 0.95 0.95];
l.FontSize = 18;
l.EdgeColor = [1 1 1];
l.NumColumns = 1;

set(gcf,'color','w');
box on
print(gcf, fullfile(figure_dir, 'F3_2.svg'), '-dsvg', '-vector');

% NMC
figure('Position', [50, 50, 350, 300]);
hold on

yyaxis left
plot(sto_nmc, p_nmc, 'LineWidth',1, 'Color', [0.4000 0.4000 0.4000]);
ax = gca;
ax.XLim = [0,1];
% ax.YLim = [0,0.7];
ax.FontSize = 20;
ax.XLabel.String = 'Stoichiometry [-]';
ax.YLabel.String = 'OCP [V]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];
% ax.XGrid = 'on';
% ax.YGrid = 'on';

yyaxis right
dVdQ_nmc = movmean(dVdQ_nmc,20);
plot(sto_nmc, dVdQ_nmc, 'LineWidth',0.5, 'Color', [0.4000 0.4000 0.4000],'LineStyle','--');

valid_idx = ~isnan(dVdQ_nmc) & sto_nmc >= 0 & sto_nmc <= 1 & dVdQ_nmc >= 0;
x_fill = sto_nmc(valid_idx);
y_fill = dVdQ_nmc(valid_idx);

y_fill(y_fill>2) = 2;
% Construct closed shape for fill: [x ->, x <-], [0s ->, y <-]
fill([x_fill; flipud(x_fill)], ...
     [zeros(size(x_fill)); flipud(y_fill)], ...
     [0.1922    0.2824    0.1961], ...
     'FaceAlpha', 0.1, 'EdgeColor', 'none');
ax.YLabel.String = 'dU/dx [V]';
ax.YLim = [0,2];
ax.YColor = [0.1 0.1 0.1];

% t = title('Composite Anode Open Circuit Potential');
% t.FontSize = 24;

l = legend("OCP","dUdx");
% l.FontAngle = 'italic';
% l.Color = [0.95 0.95 0.95];
l.FontSize = 18;
l.EdgeColor = [1 1 1];
l.NumColumns = 1;

set(gcf,'color','w');
box on
print(gcf, fullfile(figure_dir, 'F3_3.svg'), '-dsvg', '-vector');


%% plot
figure('Position', [50, 50, 350, 300]);
hold on;

% Plot the line for reference
plot(dQdV_Si, p_Si, 'LineWidth',0.5, 'Color', [0 0.4471 0.7412]);

% Define the polygon for the fill
x_fill = [0; dQdV_Si; 0];  % Start at x=0, follow x-data, and return to x=0
y_fill = [p_Si(1); p_Si; p_Si(end)]; % Start and close with the first y-data value
hold on
% Fill the area
fill(x_fill, y_fill, [0.2588    0.5059    0.6431], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

f = plot(dQdV_Gr,pGr_dQdV, 'LineWidth',0.5, 'Color', [0.5 0.5 0.5]);

% Define the polygon for the fill
x_fill = [0; dQdV_Gr; 0];  % Start at x=0, follow x-data, and return to x=0
y_fill = [pGr_dQdV(1); pGr_dQdV; pGr_dQdV(end)]; % Start and close with the first y-data value
hold on
% Fill the area
fill(x_fill, y_fill, [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.3);


ax = gca;
% ax.XLim = [0,max(Qin)];
ax.YLim = [0,0.7];ax.XLim = [-40,10];
ax.FontSize = 20;
ax.XLabel.String = 'dx/dU [1/V]';
ax.YLabel.String = 'Potential [V]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];
% ax.XGrid = 'on';
% ax.YGrid = 'on';

xticks([-40 -30 -20 -10 0 10 20]); % Set the positions of the ticks
xticklabels({'40', '30', '20', '10', '0', '10', '20'}); % Set the labels for the ticks

box on
% t = title('Effective Capacity Range of Si and Gr');
% t.FontSize = 24;
set(gcf,'color','w');

print(gcf, fullfile(figure_dir, 'F3_4.svg'), '-dsvg', '-vector');

%%

figure('Position', [50, 50, 550, 300]);
hold on
plot(sto_Si, p_Si, 'LineWidth',0.5, 'Color', [0.2588    0.5059    0.6431]);
plot(sto_Gr, p_Gr, 'LineWidth',0.5, 'Color', [0.5 0.5 0.5]);
plot(sto_Anode, Vf_anode, 'LineWidth',0.5, 'Color', [0.7608 0.3882 0.2275]);
plot(sto_Anode2, Vf_anode2, 'LineWidth',0.5, 'Color', [0.7608 0.3882 0.2275],'LineStyle','--');
plot(sto_Anode8, Vf_anode8, 'LineWidth',0.5, 'Color', [0.7608 0.3882 0.2275],'LineStyle','-.');

ax = gca;
ax.XLim = [0,1];
ax.YLim = [0,0.7];
ax.FontSize = 20;
ax.XLabel.String = 'Stoichiometry [-]';
ax.YLabel.String = 'Potential [V]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];
% ax.XGrid = 'on';
% ax.YGrid = 'on';

% t = title('Composite Anode Open Circuit Potential');
% t.FontSize = 24;

% l = legend("Silicon","Graphite","Anode (C_{Si} : C_{Gr} = 1:1 )");
% % l.FontAngle = 'italic';
% % l.Color = [0.95 0.95 0.95];
% l.FontSize = 18;
% l.EdgeColor = [1 1 1];
% l.NumColumns = 1;
box on

set(gcf,'color','w');
print(gcf, fullfile(figure_dir, 'F3_5.svg'), '-dsvg', '-vector');

%% Cell Level
% figure('Position', [50, 50, 670, 320]);
figure('Position', [50, 50, 570, 320]);
yyaxis left
f = plot(Qdata,Vfit,"LineWidth",0.5,"Color",[0.4980 0.6941 0.8000],"LineStyle","-");
hold on
f = plot(Qdata,Vfit_Cathode,"LineWidth",0.5,"Color",[0.1922    0.2824    0.1961],"LineStyle","-");

ax = gca;
ax.FontSize = 20;
ax.XLabel.String = 'Q [Ah]';
ax.YLabel.String = 'Cell V & Cathode OCP [V]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];

yyaxis right
f = plot(Qdata,Vfit_Anode,"LineWidth",0.5,"Color",[0.7608 0.3882 0.2275],"LineStyle","-");

ax = gca;
ax.XLim = [-0.25,2.7];
ax.FontSize = 20;
ax.XLabel.String = 'Q [Ah]';
ax.YLabel.String = 'Anode OCP [V]';
ax.YColor = [0.7608 0.3882 0.2275];
set(gcf,'color','w');

% add 0.2
yyaxis left
f = plot(Qdata2,Vfit2,"LineWidth",0.5,"Color",[0.4980 0.6941 0.8000],"LineStyle",":");
hold on

yyaxis right
f = plot(Qdata2,Vfit_Anode2,"LineWidth",0.5,"Color",[0.7608 0.3882 0.2275],"LineStyle",":");

% add 0.8
yyaxis left
f = plot(Qdata8,Vfit8,"LineWidth",0.5,"Color",[0.4980 0.6941 0.8000],"LineStyle","-.");
hold on
yyaxis right
f = plot(Qdata8,Vfit_Anode8,"LineWidth",0.5,"Color",[0.7608 0.3882 0.2275],"LineStyle","-.");

print(gcf, fullfile(figure_dir, 'F3_6.svg'), '-dsvg', '-vector');

%% Cell Level dQdV
figure('Position', [50, 50, 350, 300]);
hold on
plot(Qdata, -dVdQ_full, 'Color', [0.1 0.1 0.1], 'LineWidth', 0.5, 'LineStyle', '-');

f = plot(Qdata,dVdQ_Anode);
f.LineStyle = "-";
f.LineWidth = 0.5;
f.Color = [0.7608 0.3882 0.2275];

ax = gca;
xlim([-0.194, 2.5]);
ax.YLim = [0,0.8];
ax.FontSize = 20;
ax.XLabel.String = 'Q [Ah]';
ax.YLabel.String = 'dVdQ [VAh-1]';
ax.XColor = [0.1 0.1 0.1];
ax.YColor = [0.1 0.1 0.1];
set(gcf,'color','w');

% add 0.2
plot(Qdata2, -dVdQ_full2, 'Color', [0.1 0.1 0.1], 'LineWidth', 0.5, 'LineStyle', ':');

f = plot(Qdata2,dVdQ_Anode2);
f.LineStyle = ":";
f.LineWidth = 0.5;
f.Color = [0.7608 0.3882 0.2275];

% add 0.8
plot(Qdata8, -dVdQ_full8, 'Color', [0.1 0.1 0.1], 'LineWidth', 0.5, 'LineStyle', '-.');

f = plot(Qdata8,dVdQ_Anode8);
f.LineStyle = "-.";
f.LineWidth = 0.5;
f.Color = [0.7608 0.3882 0.2275];

box on

print(gcf, fullfile(figure_dir, 'F3_7.svg'), '-dsvg', '-vector');
