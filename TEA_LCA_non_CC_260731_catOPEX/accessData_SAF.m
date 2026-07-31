function processData = accessData_SAF(processData, h)
%% Units
unitCount = h.Tree.FindNode('\Data\Blocks').Elements.Count;
blockAddr = h.Tree.FindNode('\Data\Blocks');

for i = 1:unitCount
    processData.Units{i,1} = blockAddr.Elements.ItemName(i-1);
end

% Add in 'NAME' and 'TYPE' to adhere to the codes in CAPEX/OPEX evaluation.
processData.Units{unitCount+1,1} = 'DRYER1';
processData.Units{unitCount+1,15} = 'Dryer';

for i = 1:unitCount

    try % temperature
        processData.Units{i,2} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\B_TEMP']).value; %[K]
    catch
        processData.Units{i,2} = []; %[K]
    end
    try % pressure
        processData.Units{i,3} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\B_PRES']).value; %[bar]
    catch
        try % heat HX
            processData.Units{i,3} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\HOTINP']).value;
        catch
            processData.Units{i,3} = [];
        end
    end
    try % Heat duty
        processData.Units{i,4} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\QCALC']).value; %[W]
    catch
        try % heat HX
            processData.Units{i,4} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\HX_DUTY']).value;
        catch
            processData.Units{i,4} = [];
        end
    end
    try % Condenser duty
        processData.Units{i,5} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\COND_DUTY']).value; %[W]
    catch
        processData.Units{i,5} = [];
    end
    try % Reboiler duty
        processData.Units{i,6} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\REB_DUTY']).value; %[W]
    catch
        processData.Units{i,6} = [];
    end

    try % Work
        processData.Units{i,7} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\WNET']).value; %[W]
    catch
        processData.Units{i,7} = [];
    end

    try % Balance mole-flow in
        processData.Units{i,8} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\BAL_MOLI_TFL']).value; %[kmol/s]
    catch
        processData.Units{i,8} = [];
    end
    try % Balance mole-flow out
        processData.Units{i,9} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\BAL_MOLO_TFL']).value; %[kmol/s]
    catch
        processData.Units{i,9} = [];
    end
    try % Balance mass-flow in
        processData.Units{i,10} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\BAL_MASI_TFL']).value; %[kmol/s]
    catch
        processData.Units{i,10} = [];
    end
    try % Balance mass-flow out
        processData.Units{i,11} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\BAL_MASO_TFL']).value; %[kmol/s]
    catch
        processData.Units{i,11} = [];
    end
    try % Balance Enthalpy in
        processData.Units{i,12} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\TOT_ENTH_ABS']).value; %[W]
    catch
        processData.Units{i,12} = [];
    end
    try % Balance Enthalpy out
        processData.Units{i,13} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Output\BAL_ENTH_OUT']).value; %[W]
    catch
        processData.Units{i,13} = [];
    end

    numCon = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Connections']).Elements.Count;
    temp = {};
    for j = 1:numCon % Connection
        temp{1,j} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Connections']).Elements.ItemName(j-1);
        temp{2,j} = h.Tree.FindNode(['\Data\Blocks\',processData.Units{i,1},'\Connections\',temp{1,j}]).value;
    end
    processData.Units{i,14} = temp;

    processData.Units{i,15} = blockAddr.FindNode(processData.Units{i,1}).AttributeValue(6);

end

processData.Units = cell2table(processData.Units,...
    'VariableNames',{'NAME' 'TEMPERATURE' 'PRESSURE' 'HEATDUTY' 'CONDDUTY' 'REBDUTY' 'WORK' 'MOLEFLOW_IN' 'MOLEFLOW_OUT' 'MASSFLOW_IN' 'MASSFLOW_OUT' 'ENTH_IN' 'ENTH_OUT' 'CONNECTION' 'TYPE'});

%% Stream Results
allStreams = h.Tree.FindNode("\Data\Streams");
numStreams = allStreams.Elements.Count;
for i = 1:numStreams
    processData.Streams{i,1} = allStreams.Elements.ItemName(i-1);
end

for i = 1:numStreams
    try
        processData.Streams{i,2} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\TEMP_OUT\MIXED']).value; %temperature
        processData.Streams{i,3} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\PRES_OUT\MIXED']).value; %pressure
        processData.Streams{i,4} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MOLEFLMX\MIXED']).value; %total volume flow [m3/s]
        processData.Streams{i,5} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLMX\MIXED']).value; %mass density [kg/m3]
        processData.Streams{i,6} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\VOLFLMX\MIXED']).value; %total volume flow [m3/s]
        processData.Streams{i,7} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\RHOMX_MASS\MIXED']).value; %mass density [kg/m3]
        processData.Streams{i,8} = [];
        processData.Streams{i,9} = [];
        processData.Streams{i,10} = [];
        processData.Streams{i,11} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLOW\NC\BIOMASS']).value; % MASS FLOW OF BIOMASS [

    catch
        processData.Streams{i,2} = [];
        processData.Streams{i,3} = [];
        processData.Streams{i,4} = [];
        processData.Streams{i,5} = [];
        processData.Streams{i,6} = [];
        processData.Streams{i,7} = [];
        processData.Streams{i,8} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\QCALC']).value;
        processData.Streams{i,9} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\TBEGIN']).value;
        processData.Streams{i,10} = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\TEND']).value;
        processData.Streams{i,11} = [];

    end
end

disp('Accessing Data...');

processData.Streams = cell2table(processData.Streams,...
    'VariableNames',{'NAME' 'TEMPERATURE' 'PRESSURE' 'MOLEFLOW' 'MASSFLOW' 'VOLFLOW' 'DENSITY' 'DUTY' 'T_begin' 'T_end' 'BIOMASS'});

%% Input and output
% numStreams;
for i = 1:numStreams
    % all in kg/hr
    if strcmp(processData.Streams{i,1},'103') % Wood feedstock
        processData.Input.WF.Massflow = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLOW\NC\BIOMASS']).value;
    elseif strcmp(processData.Streams{i,1},'409') % Natural gas
        processData.Input.NG.Massflow = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLOW\MIXED\CH4']).value;
        disp(processData.Input.NG.Massflow)
    elseif strcmp(processData.Streams{i,1},'408') % Steam
        % processData.Input.STEAM.Massflow = h.Tree.FindNode("\Data\Streams\408\Output\MASSFLOW\MIXED\WATER").value - h.Tree.FindNode("\Data\Streams\335\Output\MASSFLOW\MIXED\WATER").value;
        processData.Input.STEAM.Massflow = max( h.Tree.FindNode("\Data\Streams\408\Output\MASSFLOW\MIXED\WATER").value - h.Tree.FindNode("\Data\Streams\335\Output\MASSFLOW\MIXED\WATER").value , 0 );

        % output from here
    elseif strcmp(processData.Streams{i,1}, '326') % SAF
        processData.Output.SAF.Massflow = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLMX\MIXED']).value;
    elseif strcmp(processData.Streams{i,1}, '324') % Gasoline
        processData.Output.Gasoline.Massflow = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLMX\MIXED']).value;
    elseif strcmp(processData.Streams{i,1},'CO2')
        % processData.Output.CO2E.Massflow = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLOW\MIXED\CO2']).value;
        % processData.Output.WASTE_W.Massflow = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLOW\MIXED\CO2']).value;
        % processData.Output.CO2_FP.Massflow = h.Tree.FindNode(['\Data\Streams\',char(processData.Streams{i,1}),'\Output\MASSFLOW\MIXED\CO2']).value;

    end
end

%add
processData.Output.H2_Prod = h.Tree.FindNode("\Data\Streams\419\Output\MASSFLMX\MIXED").value - h.Tree.FindNode("\Data\Streams\201\Output\MASSFLMX\MIXED").value;
processData.Output.H2_Consumption = h.Tree.FindNode("\Data\Streams\206\Output\MASSFLOW\MIXED\H2").value - h.Tree.FindNode("\Data\Streams\215\Output\MASSFLOW\MIXED\H2").value;
processData.Output.CO2E.Massflow = h.Tree.FindNode(['\Data\Streams\424\Output\MASSFLOW\MIXED\CO2']).value;
processData.Output.WASTE_W.Massflow = h.Tree.FindNode("\Data\Streams\327\Output\MASSFLMX\MIXED").value;
% processData.Output.CO2_FP1.Massflow = h.Tree.FindNode("\Data\Streams\142\Output\MASSFLOW\MIXED\CO2").value;
% processData.Output.CO2_FP2.Massflow = h.Tree.FindNode("\Data\Streams\107\Output\MASSFLOW\MIXED\CO2").value;
% processData.Output.CO2_FR.Massflow = h.Tree.FindNode("\Data\Streams\330\Output\MASSFLOW\MIXED\CO2").value;
% processData.Output.CO2_CO2CAPTURE.Massflow = h.Tree.FindNode("\Data\Streams\426\Output\MASSFLOW\MIXED\CO2").value;

end