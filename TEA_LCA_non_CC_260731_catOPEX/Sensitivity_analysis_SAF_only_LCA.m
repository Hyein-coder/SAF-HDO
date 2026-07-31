simFolder = "C:\Users\user\Desktop\SAF_HDO\res_20260403_non_CC\";

list_simulations = {
    '260401_pyrolysis_oil_CC_case_j_HOadjustment3_non_CC.bkp';
    '260401_pyrolysis_oil_CC_case_k_6_non_CC.bkp';
    '260120_pyrolysis_oil_CC_case_i_non_CC.bkp';
    };

nSim = length(list_simulations);
outputFile = fullfile(simFolder,'SAF_Sensitivity_Analysis_260407_non_CC_all_LCA.xlsx');

%% LCA
Results = table();
Res_ME_BALANCE = table();

for idxSim = 1:nSim

    pfad = fullfile(simFolder, list_simulations{idxSim});
    fprintf('\nProcessing simulation (LCA): %s\n', pfad);

    h = actxserver('Apwn.Document');
    invoke(h,'InitFromArchive2', pfad);
    h.Engine.Reinit;
    h.Engine.Run2;

    processData = struct();
    processData.LCA = struct();
    f = {'elec','steam','ref','WF','NG'};
    for ii = 1:length(f)
        processData.LCA.(f{ii}) = [];
    end

    processData = accessData_SAF(processData, h);
    unitCount   = size(processData.Units,1);

    COST_PARAMETER;
    LCA_PARAMETER;

    processData = CAPEX_SAF(processData, unitCount, COST, h);
    processData = OPEX_SAF(processData, unitCount, COST, h);
    processData = cash_flow_generic_SAF(processData, COST);
    processData = LCA_analysis_SAF(processData, LCA);

    fun_msp = @(x)msp_SAF(processData, COST, x);
    msp_LCA = fsolve(fun_msp, 3);
    processData_msp = cash_flow_generic_SAF_opt(processData, COST, msp_LCA);

    Read_results;

    row = table();
    row.Case = string(list_simulations{idxSim});
%%
    % Allocation Gasoline
    if isfield(processData_msp,'LCA_alloc_Gasoline')
        allocGas = processData_msp.LCA_alloc_Gasoline(:)';
    else
        allocGas = NaN;
    end
    row = addVectorDynamic(row,'Alloc_Gasoline',allocGas);

    % Allocation SAF
    if isfield(processData_msp,'LCA_alloc_SAF')
        allocSAF = processData_msp.LCA_alloc_SAF(:)';
    else
        allocSAF = NaN;
    end
    row = addVectorDynamic(row,'Alloc_SAF',allocSAF);

    % Electricity LCA
    elecTypes = {'Grid','Future','Nuclear','Wind','PV'};
    for iType = 1:length(elecTypes)
        typeName = elecTypes{iType};
        vec = [];

        if ~isempty(processData_msp.LCA.elec)
            if iscell(processData_msp.LCA.elec)
                if length(processData_msp.LCA.elec) >= iType
                    vec = processData_msp.LCA.elec{iType};
                end
            else
                if size(processData_msp.LCA.elec,2) >= iType
                    vec = processData_msp.LCA.elec(:,iType)';
                else
                    vec = processData_msp.LCA.elec(:)';
                end
            end
        end

        if isempty(vec), vec = NaN; end
        prefix = sprintf('Elec_%s', typeName);
        row = addVectorDynamic(row, prefix, vec);
    end

    % Steam LCA
    steamVec = [];
    if isfield(processData_msp.LCA,'steam') && ~isempty(processData_msp.LCA.steam)
        steamVec = processData_msp.LCA.steam(:)';
    end
    if isempty(steamVec), steamVec = NaN; end
    row = addVectorDynamic(row,'Steam',steamVec);

    % Refrigerant LCA
    refVec = [];
    if isfield(processData_msp.LCA,'ref') && ~isempty(processData_msp.LCA.ref)
        refVec = processData_msp.LCA.ref(:)';
    end
    if isempty(refVec), refVec = NaN; end
    row = addVectorDynamic(row,'Ref',refVec);

    % WWT
    wwtVec = [];
    if isfield(processData_msp,'Output') && ...
            isfield(processData_msp.Output,'WASTE_W') && ...
            isfield(processData_msp.Output.WASTE_W,'Massflow')

        wwtVec = processData_msp.Output.WASTE_W.Massflow(:)';
    end
    if isempty(wwtVec), wwtVec = NaN; end
    row = addVectorDynamic(row,'WWT',wwtVec);

    % Biomass (WF - Wood Feedstock)
    wfVec = [];
    if isfield(processData_msp.LCA,'WF') && ~isempty(processData_msp.LCA.WF)
        wfVec = processData_msp.LCA.WF(:)';
    end
    if isempty(wfVec), wfVec = NaN; end
    row = addVectorDynamic(row,'Biomass',wfVec);

    % Natural Gas (NG)
    ngVec = [];
    if isfield(processData_msp.LCA,'NG') && ~isempty(processData_msp.LCA.NG)
        ngVec = processData_msp.LCA.NG(:)';
    end
    if isempty(ngVec), ngVec = NaN; end
    row = addVectorDynamic(row,'NG',ngVec);

    %% LCA Scenario
    if isfield(processData_msp.LCA,'scenario')
        sc = processData_msp.LCA.scenario;
        scFields = fieldnames(sc);

        for f = 1:length(scFields)
            fieldName = scFields{f};
            data = sc.(fieldName);

            if isnumeric(data)
                vec = data(:)';
                if isempty(vec), vec = NaN; end
                prefix = sprintf('Scenario_%s', fieldName);
                row = addVectorDynamic(row, prefix, vec);

            elseif isstruct(data)
                subFields = fieldnames(data);
                for sf = 1:length(subFields)
                    subVec = data.(subFields{sf})(:)';
                    if isempty(subVec), subVec = NaN; end
                    prefix = sprintf('Scenario_%s_%s', fieldName, subFields{sf});
                    row = addVectorDynamic(row, prefix, subVec);
                end
            end
        end
    end

    release(h);
    LCA_mass_energy_balance
    me_bal_table = struct2table(me_balance);

    if isempty(Results)
        Results = row;
    else
        newColsInRow = setdiff(row.Properties.VariableNames, Results.Properties.VariableNames);
        for nc = 1:length(newColsInRow)
            Results.(newColsInRow{nc}) = NaN(height(Results),1);
        end

        newColsInResults = setdiff(Results.Properties.VariableNames, row.Properties.VariableNames);
        for nc = 1:length(newColsInResults)
            row.(newColsInResults{nc}) = NaN;
        end

        row = row(:, Results.Properties.VariableNames);

        Results = [Results; row];
    end

    if isempty(Res_ME_BALANCE)
        Res_ME_BALANCE = me_bal_table;
    else
        newColsInRow = setdiff(me_bal_table.Properties.VariableNames, Res_ME_BALANCE.Properties.VariableNames);
        for nc = 1:length(newColsInRow)
            Res_ME_BALANCE.(newColsInRow{nc}) = NaN(height(Res_ME_BALANCE),1);
        end

        newColsInResults = setdiff(Res_ME_BALANCE.Properties.VariableNames, me_bal_table.Properties.VariableNames);
        for nc = 1:length(newColsInResults)
            me_bal_table.(newColsInResults{nc}) = NaN;
        end

        me_bal_table = me_bal_table(:, Res_ME_BALANCE.Properties.VariableNames);

        Res_ME_BALANCE = [Res_ME_BALANCE; me_bal_table];
    end

end

%% TO EXCEL
fprintf('\n=== LCA results to Excel ===\n');

try
    writetable(Results, outputFile, 'Sheet', 'LCA', 'WriteRowNames', false);
catch ME
    warning('Failed to write LCA sheet: %s', ME.message);
end

try 
    writetable(Res_ME_BALANCE, outputFile, 'Sheet', 'ME_BAL', 'WriteRowNames', false);
catch ME
    warning('Failed to write ME_BAL sheet: %s', ME.message);
end 

function row = addVectorDynamic(row, prefix, vec)
if nargin < 3 || isempty(vec)
    vec = NaN;
end

vec = vec(:)';

for j = 1:length(vec)
    colName = sprintf('%s_%d', prefix, j);
    row.(colName) = vec(j);
end
end