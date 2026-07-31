clear;
clc;

pfad = "C:\Users\user\Desktop\SAF_HDO\forMATLAB\Simulation\260115_pyrolysis_oil_CC_case_a$backup.bkp";
% pfad = 'D:\523506\A\Create_251204_SA\Simulation\260120_pyrolysis_oil_CC_case_i_run.bkp';
% pfad = 'D:\523506\A\Create_251204_SA\Simulation\260115_pyrolysis_oil_CC_case_a_run.bkp';
% pfad = 'D:\523506\A\Create_251204_SA\Simulation\251204_pyrolysis_oil_CC_case_i_run.bkp';
% pfad = 'D:\523506\A\Create_251120_SA\Simulation\251125_basecase_i_2_run.bkp';
% pfad = 'D:\523506\A\Create_251120_SA\Simulation\251125_basecase_a_run.bkp';
% pfad = 'D:\523506\A\Create_251120_SA\Simulation\251125_basecase_a_Copy_run.bkp';
% pfad = 'D:\523506\A\Create_251120_SA\Simulation\sensitivity_20251121\grid_20251121_173949_param_0\converged\caseSA_2.bkp';
% pfad = 'D:\523506\A\Create_251120_SA\Simulation\251122_pyrolysis_oil_CC_case_a_run.bkp';
% "D:\523506\A\Create_251120_SA\Simulation\sensitivity_20251121\grid_20251121_173949_param_0\converged\caseSA_2.bkp"
% pfad = ['D:\523506\A\Create_251104_2\Simulation\caseK_1.0_run.bkp'];

% pfad = 'D:\523506\A\Create_251104_2\Simulation\251112_pyrolysis_oil_CC_case_a_run.bkp';
% pfad = 'D:\523506\A\Create_251104_2\Simulation\251112_pyrolysis_oil_CC_case_k_run.bkp';
% pfad = 'D:\523506\A\Create_251104_2\Simulation\251111_pyrolysis_oil_CC_case_i_run.bkp';

% pfad = 'D:\523506\A\Create_250917\Simulation\251023_pyrolysis_oil_CC_case_k_run.bkp';
% pfad = 'D:\523506\A\Create_250917\Simulation\251023_pyrolysis_oil_CC_case_a_run.bkp';
% pfad = 'D:\523506\A\Create_250917\Simulation\251022_pyrolysis_oil_CC_case_k_run.bkp';
% pfad = 'D:\523506\A\Create_250917\Simulation\251022_pyrolysis_oil_CC_case_a_run.bkp';
% pfad = 'D:\523506\A\Create_250917\Simulation\251009_pyrolysis_oil_CC_3_run.bkp';
% pfad = 'D:\523506\A\Create_250917\Simulation\250918_pyrolysis_oil_CC_run.bkp';
h = actxserver('Apwn.Document');    %active X handle
invoke(h,'InitFromArchive2', pfad);

%% Block for testing
h.Engine.Reinit;
h.Engine.Run2;
Cov_State = h.Tree.FindNode("\Data\Results Summary\Run-Status\Output\UOSSTAT2").value;

%% Block
clear processData
processData = [];

processData.LCA.elec = [];
processData.LCA.steam = [];
processData.LCA.ref = [];
processData.LCA.WF = []; % wood feedstock
processData.LCA.NG = []; % natural gas

processData = accessData_SAF(processData, h);
unitCount = size(processData.Units,1);

%% Unit Cost Estimation
processData.Units = addvars(processData.Units, zeros(unitCount,1));
processData.Units.Properties.VariableNames(end) = {'CAPEX'};
processData.Units = addvars(processData.Units, zeros(unitCount,1));
processData.Units.Properties.VariableNames(end) = {'WORK_OPEX'};
processData.Units = addvars(processData.Units, zeros(unitCount,1));
processData.Units.Properties.VariableNames(end) = {'STEAM_OPEX'};
processData.Units = addvars(processData.Units, zeros(unitCount,1));
processData.Units.Properties.VariableNames(end) = {'REF_OPEX'};
processData.Units = addvars(processData.Units, zeros(unitCount,1));
processData.Units.Properties.VariableNames(end) = {'Size'};
processData.Units = addvars(processData.Units, zeros(unitCount,1));
processData.Units.Properties.VariableNames(end) = {'Diameter'};
processData.Units = addvars(processData.Units, zeros(unitCount,1));
processData.Units.Properties.VariableNames(end) = {'Height'};

%% Calculation for initial input materials
COST_PARAMETER
LCA_PARAMETER

processData.LCA.elec = [0 0 0];

%% Cash Flow
processData = CAPEX_SAF(processData, unitCount, COST, h);
processData = OPEX_SAF(processData, unitCount, COST, h);
processData = cash_flow_generic_SAF(processData, COST);
processData = LCA_analysis_SAF(processData, LCA);

y_NPV = processData.NPV;
% save(filename, 'y_NPV', 'processData');

%% Minimum Selling Price
fun = @(x)msp_SAF(processData, COST, x);
x0 = [3];
msp = fsolve(fun, x0);

processData_msp = [];
% processData_msp = cash_flow_generic_SAF_opt(processData, COST, msp);
processData_msp = cash_flow_generic_SAF_opt_mjsp(processData, COST, msp);

Read_results;

%% Quit simulation
release(h);