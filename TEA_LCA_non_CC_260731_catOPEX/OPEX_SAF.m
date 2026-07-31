function processData = OPEX_SAF(processData, unitCount, COST, h)
%% Electricity Cost
% Electrolyzer
OperatingCost.ELECTRICITY = [];
OperatingCost.ELECTRICITY = 0;          % $/kW-hr * kWh = $/year
OperatingCost.ELECTRICITY = [OperatingCost.ELECTRICITY, 0];
% Compressor and Pump
% OperatingCost.ELECTRICITY
for i = 1:unitCount
    if strcmp(processData.Units.TYPE{i},'Compr') || strcmp(processData.Units.TYPE{i}, 'Pump')
        Work = processData.Units.WORK{i}; % [kW]
        WorkTotal = Work * COST.utility.electricity * COST.utility.operating_hour; % kWh/year
        OperatingCost.ELECTRICITY = [OperatingCost.ELECTRICITY WorkTotal]; % $/year
        disp('el')
        disp(processData.Units.NAME{i})
        processData.LCA.elec = [processData.LCA.elec, Work];
    end
end

%% STEAM & Refrigerant Cost
% Distillation Column ($)
OperatingCost.STEAM = [];
OperatingCost.REFRIGERATION = [];
for i = 1:unitCount
    if strcmp(processData.Units.TYPE{i},'RadFrac')
        % Reboiler
        cost_reboiler = 0;
        cost_condenser = 0;
        numStage = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Input\NSTAGE']).value / 2;
        feedtemp = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\B_TEMP\',num2str(numStage)]).value + 273.15; % [K]
        bottomtemp = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\BOTTOM_TEMP']).value + 273.15; % [K]
        heat_duty = abs(processData.Units.REBDUTY{i})*4.1868; % [cal/s] to [J/s]

        if bottomtemp <=147.778 + 273.15
            cost_reboiler = COST.utility.steam50psig * heat_duty *3600 * COST.utility.operating_hour;
        elseif 147.778 + 273.15<=bottomtemp && bottomtemp <=185.556 + 273.15
            cost_reboiler = COST.utility.steam150psig * heat_duty *3600 * COST.utility.operating_hour;
        elseif 185.556 + 273.15<=bottomtemp && bottomtemp <=237.778 + 273.15
            cost_reboiler = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
        elseif 237.778 + 273.15<=bottomtemp
            cost_reboiler = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
        else
            cost_reboiler = 7 * 10^(-7) * (heat_duty / 1000)^(-0.9) * 500^(0.5) *603 + 6 * 10^(-8)*500^(0.5)*9;
        end

        % Condenser
        toptemp = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\TOP_TEMP']).value + 273.15; % [K]
        heat_dutyc = abs(processData.Units.CONDDUTY{i})*4.1868; % [J/s]
        if toptemp <= -101.111 + 273.15
            cost_condenser = COST.utility.Rn150 * heat_dutyc *3600 * COST.utility.operating_hour;
            disp('infeasible temperature (CONDENSER) check the steam');
        elseif -101.111 + 273.15<=toptemp && toptemp <= -67.7778 + 273.15
            cost_condenser = COST.utility.Rn90 * heat_dutyc *3600 * COST.utility.operating_hour;
        elseif -67.7778 + 273.15<=toptemp && toptemp <= -34.4444 + 273.15
            cost_condenser = COST.utility.Rn30 * heat_dutyc *3600 * COST.utility.operating_hour;
        elseif -34.4444 + 273.15<=toptemp && toptemp <= -12.2222 + 273.15
            cost_condenser = COST.utility.Rp10 * heat_dutyc *3600 * COST.utility.operating_hour;
        elseif -12.2222 + 273.15<=toptemp && toptemp <= 4.4444 + 273.15
            cost_condenser = COST.utility.Rchw * heat_dutyc *3600 * COST.utility.operating_hour;
        elseif 4.4444 + 273.15<=toptemp && toptemp <= 25 + 273.15
            cost_condenser = COST.utility.cw * heat_dutyc *3600 * COST.utility.operating_hour;
        elseif 25 + 273.15<=toptemp
            cost_condenser = COST.utility.cw * heat_dutyc *3600 * COST.utility.operating_hour;
        else
            cost_condenser = COST.utility.cw/10 * heat_dutyc *3600 * COST.utility.operating_hour;
        end
        % cost
        OperatingCost.STEAM = [OperatingCost.STEAM cost_reboiler];
        disp('st')
        disp(processData.Units.NAME{i})
        OperatingCost.REFRIGERATION = [OperatingCost.REFRIGERATION cost_condenser];
        disp('re')
        disp(processData.Units.NAME{i})
        processData.LCA.steam = [processData.LCA.steam heat_duty*3600/processData.Output.SAF.Massflow/1000000];
        processData.LCA.ref = [processData.LCA.ref heat_dutyc*3600/processData.Output.SAF.Massflow/1000000];
    end
end

% Heater ($)
for i = 1:unitCount
    % if strcmp(processData.Units.NAME{i}, 'COOLER-3')
    %     stname='S23';
    %     T1 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]
    %     stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
    %     T2 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]
    %     heat_duty = abs(processData.Units.HEATDUTY{i})*4.1868; % [J/s]
    %
    %     if T1-T2 > 0
    %         % cooling
    %         if T2 <= -101.111 + 273.15
    %             cost_heater = COST.utility.Rn150 * heat_duty *3600 * COST.utility.operating_hour;
    %             disp('infeasible temperature (CONDENSER) check the steam');
    %         elseif -101.111 + 273.15<=T2 && T2 <= -67.7778 + 273.15
    %             cost_heater = COST.utility.Rn90 * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif -67.7778 + 273.15<=T2 && T2 <= -34.4444 + 273.15
    %             cost_heater = COST.utility.Rn30 * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif -34.4444 + 273.15<=T2 && T2 <= -12.2222 + 273.15
    %             cost_heater = COST.utility.Rp10 * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif -12.2222 + 273.15<=T2 && T2 <= 4.4444 + 273.15
    %             cost_heater = COST.utility.Rchw * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 4.4444 + 273.15<=T2 && T2 <= 25 + 273.15
    %             cost_heater = COST.utility.cw * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 25 + 273.15<=T2
    %             cost_heater = COST.utility.cw/10 * heat_duty *3600 * COST.utility.operating_hour;
    %         end
    %         % cost
    %         OperatingCost.REFRIGERATION = [OperatingCost.REFRIGERATION cost_heater];
    %         disp('re')
    %         disp(processData.Units.NAME{i})
    %         processData.LCA.ref = [processData.LCA.ref heat_duty*3600/processData.Output.SAF.Massflow/1000000];
    %
    %     elseif T1-T2 < 0
    %         % heating
    %         if T2 <=147.778 + 273.15
    %             cost.heater = COST.utility.steam50psig * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 147.778 + 273.15<=T2 && T2 <=185.556 + 273.15
    %             cost.heater = COST.utility.steam150psig * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 185.556 + 273.15<=T2 && T2 <=237.778 + 273.15
    %             cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 237.778 + 273.15<=T2
    %             cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
    %         else
    %             cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
    %         end
    %         % cost
    %         OperatingCost.STEAM = [OperatingCost.STEAM cost_heater];
    %         disp('st')
    %         disp(processData.Units.NAME{i})
    %         processData.LCA.steam = [processData.LCA.steam heat_duty*3600/processData.Output.SAF.Massflow/1000000];
    %     end
    % end
    %
    % % code appended from line 136 to 185, for calculating heat duty of final reactor.
    % if strcmp(processData.Units.NAME{i}, 'EC-REAC')
    %
    %     stname='SOL-H2';
    %     T1 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]
    %     stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
    %     T2 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]
    %     heat_duty = abs(processData.Units.HEATDUTY{i})*4.1868; % [J/s]
    %
    %     if T1-T2 > 0
    %         % cooling
    %         if T2 <= -101.111 + 273.15
    %             cost_heater = COST.utility.Rn150 * heat_duty *3600 * COST.utility.operating_hour;
    %             disp('infeasible temperature (CONDENSER) check the steam');
    %         elseif -101.111 + 273.15<=T2 && T2 <= -67.7778 + 273.15
    %             cost_heater = COST.utility.Rn90 * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif -67.7778 + 273.15<=T2 && T2 <= -34.4444 + 273.15
    %             cost_heater = COST.utility.Rn30 * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif -34.4444 + 273.15<=T2 && T2 <= -12.2222 + 273.15
    %             cost_heater = COST.utility.Rp10 * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif -12.2222 + 273.15<=T2 && T2 <= 4.4444 + 273.15
    %             cost_heater = COST.utility.Rchw * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 4.4444 + 273.15<=T2 && T2 <= 25 + 273.15
    %             cost_heater = COST.utility.cw * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 25 + 273.15<=T2
    %             cost_heater = COST.utility.cw/10 * heat_duty *3600 * COST.utility.operating_hour;
    %         end
    %         % cost
    %         OperatingCost.REFRIGERATION = [OperatingCost.REFRIGERATION cost_heater];
    %         disp('re')
    %         disp(processData.Units.NAME{i})
    %         processData.LCA.ref = [processData.LCA.ref heat_duty*3600/processData.Output.SAF.Massflow/1000000];
    %
    %      elseif T1-T2 < 0
    %         % heating
    %         if T2 <=147.778 + 273.15
    %             cost.heater = COST.utility.steam50psig * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 147.778 + 273.15<=T2 && T2 <=185.556 + 273.15
    %             cost.heater = COST.utility.steam150psig * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 185.556 + 273.15<=T2 && T2 <=237.778 + 273.15
    %             cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
    %         elseif 237.778 + 273.15<=T2
    %             cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
    %             % disp('infeasible temperature (REBOILER) check the steam');
    %         else
    %             cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
    %         end
    %         % cost
    %         OperatingCost.STEAM = [OperatingCost.STEAM cost_heater];
    %         disp('st')
    %         disp(processData.Units.NAME{i})
    %         processData.LCA.steam = [processData.LCA.steam heat_duty*3600/processData.Output.SAF.Massflow/1000000];
    %     end
    % end

    if strcmp (processData.Units.TYPE{i},'Heater') % && ~strcmp(processData.Units.NAME{i},'COOLER-3')
        stf1=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'HS(IN)'));
        stf2=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'HS(OUT)'));

        if isempty(stf1) && isempty(stf2)
            stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'F(IN)'));
            T1 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]

            stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
            T2 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]
            heat_duty = abs(processData.Units.HEATDUTY{i})*4.1868; % [J/s]

            if T1-T2 > 0
                % cooling
                if T2 <= -101.111 + 273.15
                    cost_heater = COST.utility.Rn150 * heat_duty *3600 * COST.utility.operating_hour;
                    disp('infeasible temperature (CONDENSER) check the steam');
                elseif -101.111 + 273.15<=T2 && T2 <= -67.7778 + 273.15
                    cost_heater = COST.utility.Rn90 * heat_duty *3600 * COST.utility.operating_hour;
                elseif -67.7778 + 273.15<=T2 && T2 <= -34.4444 + 273.15
                    cost_heater = COST.utility.Rn30 * heat_duty *3600 * COST.utility.operating_hour;
                elseif -34.4444 + 273.15<=T2 && T2 <= -12.2222 + 273.15
                    cost_heater = COST.utility.Rp10 * heat_duty *3600 * COST.utility.operating_hour;
                elseif -12.2222 + 273.15<=T2 && T2 <= 4.4444 + 273.15
                    cost_heater = COST.utility.Rchw * heat_duty *3600 * COST.utility.operating_hour;
                elseif 4.4444 + 273.15<=T2 && T2 <= 25 + 273.15
                    cost_heater = COST.utility.cw * heat_duty *3600 * COST.utility.operating_hour;
                elseif 25 + 273.15<=T2
                    cost_heater = COST.utility.cw/10 * heat_duty *3600 * COST.utility.operating_hour;
                end
                % cost
                OperatingCost.REFRIGERATION = [OperatingCost.REFRIGERATION cost_heater];
                disp('re')
                disp(processData.Units.NAME{i})
                processData.LCA.ref = [processData.LCA.ref heat_duty*3600/processData.Output.SAF.Massflow/1000000];

            elseif T1-T2 < 0
                % heating
                if T2 <=147.778 + 273.15
                    cost_heater = COST.utility.steam50psig * heat_duty *3600 * COST.utility.operating_hour;
                elseif 147.778 + 273.15<=T2 && T2 <=185.556 + 273.15
                    cost_heater = COST.utility.steam150psig * heat_duty *3600 * COST.utility.operating_hour;
                elseif 185.556 + 273.15<=T2 && T2 <=237.778 + 273.15
                    cost_heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                elseif 237.778 + 273.15<=T2
                    cost_heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                    % disp('infeasible temperature (REBOILER) check the steam');
                else
                    cost_heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                end
                % cost
                OperatingCost.STEAM = [OperatingCost.STEAM cost_heater];
                disp('st')
                disp(processData.Units.NAME{i})
                processData.LCA.steam = [processData.LCA.steam heat_duty*3600/processData.Output.SAF.Massflow/1000000];
            end
        end

        if ~isempty(stf1)
            if processData.Streams.DUTY{strcmp(processData.Streams.NAME(:), stf1),1} == 0
                stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'F(IN)'));
                T1 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]

                stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
                T2 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]
                heat_duty = abs(processData.Units.HEATDUTY{i})*4.1868; % [J/s]

                if T1-T2 > 0
                    % cooling
                    if T2 <= -101.111 + 273.15
                        cost_heater = COST.utility.Rn150 * heat_duty *3600 * COST.utility.operating_hour;
                        disp('infeasible temperature (CONDENSER) check the steam');
                    elseif -101.111 + 273.15<=T2 && T2 <= -67.7778 + 273.15
                        cost_heater = COST.utility.Rn90 * heat_duty *3600 * COST.utility.operating_hour;
                    elseif -67.7778 + 273.15<=T2 && T2 <= -34.4444 + 273.15
                        cost_heater = COST.utility.Rn30 * heat_duty *3600 * COST.utility.operating_hour;
                    elseif -34.4444 + 273.15<=T2 && T2 <= -12.2222 + 273.15
                        cost_heater = COST.utility.Rp10 * heat_duty *3600 * COST.utility.operating_hour;
                    elseif -12.2222 + 273.15<=T2 && T2 <= 4.4444 + 273.15
                        cost_heater = COST.utility.Rchw * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 4.4444 + 273.15<=T2 && T2 <= 25 + 273.15
                        cost_heater = COST.utility.cw * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 25 + 273.15<=T2
                        cost_heater = COST.utility.cw/10 * heat_duty *3600 * COST.utility.operating_hour;
                    end
                    % cost
                    OperatingCost.REFRIGERATION = [OperatingCost.REFRIGERATION cost_heater];
                    disp('re')
                    disp(processData.Units.NAME{i})
                    processData.LCA.ref = [processData.LCA.ref heat_duty*3600/processData.Output.SAF.Massflow/1000000];

                elseif T1-T2 < 0
                    % heating
                    if T2 <=147.778 + 273.15
                        cost.heater = COST.utility.steam50psig * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 147.778 + 273.15<=T2 && T2 <=185.556 + 273.15
                        cost.heater = COST.utility.steam150psig * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 185.556 + 273.15<=T2 && T2 <=237.778 + 273.15
                        cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 237.778 + 273.15<=T2
                        cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                        % disp('infeasible temperature (REBOILER) check the steam');
                    else
                        cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                    end
                    % cost
                    OperatingCost.STEAM = [OperatingCost.STEAM cost_heater];
                    disp('st')
                    disp(processData.Units.NAME{i})
                    processData.LCA.steam = [processData.LCA.steam heat_duty*3600/processData.Output.SAF.Massflow/1000000];
                end
            else

                % cost_heater = 0;
                % OperatingCost.STEAM = [OperatingCost.STEAM cost_heater];
            end

        elseif ~isempty(stf2)
            if processData.Streams.DUTY{strcmp(processData.Streams.NAME(:), stf2),1} == 0
                stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'F(IN)'));
                T1 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]

                stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
                T2 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]
                heat_duty = abs(processData.Units.HEATDUTY{i})*4.1868; % [J/s]

                if T1-T2 > 0
                    % cooling
                    if T2 <= -101.111 + 273.15
                        cost_heater = COST.utility.Rn150 * heat_duty *3600 * COST.utility.operating_hour;
                        disp('infeasible temperature (CONDENSER) check the steam');
                    elseif -101.111 + 273.15<=T2 && T2 <= -67.7778 + 273.15
                        cost_heater = COST.utility.Rn90 * heat_duty *3600 * COST.utility.operating_hour;
                    elseif -67.7778 + 273.15<=T2 && T2 <= -34.4444 + 273.15
                        cost_heater = COST.utility.Rn30 * heat_duty *3600 * COST.utility.operating_hour;
                    elseif -34.4444 + 273.15<=T2 && T2 <= -12.2222 + 273.15
                        cost_heater = COST.utility.Rp10 * heat_duty *3600 * COST.utility.operating_hour;
                    elseif -12.2222 + 273.15<=T2 && T2 <= 4.4444 + 273.15
                        cost_heater = COST.utility.Rchw * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 4.4444 + 273.15<=T2 && T2 <= 25 + 273.15
                        cost_heater = COST.utility.cw * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 25 + 273.15<=T2
                        cost_heater = COST.utility.cw/10 * heat_duty *3600 * COST.utility.operating_hour;
                    end
                    % cost
                    OperatingCost.REFRIGERATION = [OperatingCost.REFRIGERATION cost_heater];
                    disp('re')
                    disp(processData.Units.NAME{i})
                    processData.LCA.ref = [processData.LCA.ref heat_duty*3600/processData.Output.SAF.Massflow/1000000];

                elseif T1-T2 < 0
                    % heating
                    if T2 <=147.778 + 273.15
                        cost.heater = COST.utility.steam50psig * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 147.778 + 273.15<=T2 && T2 <=185.556 + 273.15
                        cost.heater = COST.utility.steam150psig * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 185.556 + 273.15<=T2 && T2 <=237.778 + 273.15
                        cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                    elseif 237.778 + 273.15<=T2
                        cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                        % disp('infeasible temperature (REBOILER) check the steam');
                    else
                        cost.heater = COST.utility.steam450psig * heat_duty *3600 * COST.utility.operating_hour;
                    end
                    % cost
                    OperatingCost.STEAM = [OperatingCost.STEAM cost_heater];
                    disp('st')
                    disp(processData.Units.NAME{i})
                    processData.LCA.steam = [processData.LCA.steam heat_duty*3600/processData.Output.SAF.Massflow/1000000];
                end
            else
            end
        end
    end

    % Calculate energy usage according to latent heat of vaporization
    % (22/01/19, Overestimated)
    if strcmp(processData.Units.TYPE{i},'Dryer')
        H_vap = 2260; % [kJ/kg]
        % moisture_rate = 1413.612; % temp value
        moisture_rate = processData.Units.WORK{i};
        H_tot = H_vap * moisture_rate * 1000 / 3600; % kJ/hr to J/s
        cost_heater = COST.utility.steam50psig * H_tot * 3600 * COST.utility.operating_hour;

        OperatingCost.STEAM = [OperatingCost.STEAM cost_heater];
        disp('st')
        disp(processData.Units.NAME{i})
        processData.LCA.steam = [processData.LCA.steam heat_duty*3600/processData.Output.SAF.Massflow/1000000];
    end
end

% Carbon Capture OPEX
% G_re = 4;                     % Energy required for CO2 capture [GJ/ton]
% C_steam = 0.0066;             % Steam cost [$/kg]
% T_op = 8000;                  % Operating time [h/year]
% steam_energy_per_kg = 2.326;  % [MJ/kg]

% C_cap_op = processData.Output.CO2_CO2CAPTURE.Massflow * T_op * G_re * C_steam/steam_energy_per_kg * 1000; % GJ/MJ ---> $/year
% C_cap_op = processData.Output.CO2_CO2CAPTURE.Massflow * (G_re * 1000 / 1000) * C_steam / steam_energy_per_kg * T_op;


% 250607 ---> https://doi.org/10.1016/j.ifacol.2022.07.458
% C_opex_base = 6.105e+06;            % Base CAPEX [$]
% % CO2_base = 196000               % [t/y]
% CO2_base = 24500;               % Base flow [kg/h]
%
% C_cap_op = C_opex_base * (processData.Output.CO2_CO2CAPTURE.Massflow / CO2_base);

% C_opex_base = 40; % [$/year]

% BaseIndex_2013 = 567.3;  % CEPCI
% BaseIndex_2022 = 816; % CEPCI
% Carbon_capture_without_sulfur_removal_2013 = 44; % [$/ton CO2] https://pubs.acs.org/doi/full/10.1021/ie402185h
% EscalationFactor = BaseIndex_2022 / BaseIndex_2013;
% Carbon_capture_without_sulfur_removal = Carbon_capture_without_sulfur_removal_2013 * EscalationFactor; 
% Portion_carbon_capture_OPEX = 0.486;
% C_opex_base = Carbon_capture_without_sulfur_removal * Portion_carbon_capture_OPEX; % [$/tonne]
% C_cap_op = C_opex_base * (processData.Output.CO2_CO2CAPTURE.Massflow / 1000) * T_op;
% OperatingCost.Capture = C_cap_op;

% Catalyst OPEX
C_catalyst = 59707.35; %$
Catalyst_OPEX = (1/3) * C_catalyst;
OperatingCost.Catalyst = Catalyst_OPEX;

% Waste Water Treatment
OperatingCost.WWT = processData.Output.WASTE_W.Massflow * COST.utility.wastewt * COST.utility.operating_hour; % kg/hr * $/kg * operating hour -> $/yr

processData.OperatingCost = OperatingCost;
end
