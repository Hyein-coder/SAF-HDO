function [processData] = CAPEX_SAF(processData, unitCount, COST, h)

CapitalCosts.flash2 = [];
CapitalCosts.flash3 = [];
CapitalCosts.distillation = [];
CapitalCosts.compressor = [];
CapitalCosts.pump = [];
CapitalCosts.heater = [];
CapitalCosts.heatX = [];
CapitalCosts.reactor = [];
CapitalCosts.separator = [];
CapitalCosts.dryer = [];
CapitalCosts.PSA = [];

for i = 1:unitCount
    % Component Separation
    if strcmp(processData.Units.TYPE(i), 'Sep')
        % if strcmp(processData.Units.NAME(i), 'PSEUDO') || strcmp(processData.Units.NAME(i), 'DUM_PSA')
        processData.Units.CAPEX(i) = 0;
        % elseif strcmp(processData.Units.NAME(i), 'S-101')
        % if strcmp(processData.Units.NAME(i), 'S-101')
        %
        %     stname='107'; % Flash liquid
        %     LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1) / 3600; % [kg/s]
        %     LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        %     VesselPressure = 1; % processData.Units.PRESSURE{i}; % [bar]
        %     [co,si]= NPV_flash_2020(LFlow,LDensity,VesselPressure,COST.UPF);
        %     processData.Units.CAPEX(i) = co;
        %     processData.Units.Size(i) = si;
        %     CapitalCosts.separator = [CapitalCosts.separator, processData.Units.CAPEX(i)];

        %         if strcmp(processData.Units.NAME(i), 'S-104')
        %
        %             stname='142'; % Flash liquid
        %             LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1) / 3600; % [kg/s]
        %             LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        %             VesselPressure = 1; % processData.Units.PRESSURE{i}; % [bar]
        %             [co,si]= NPV_flash_2020(LFlow,LDensity,VesselPressure,COST.UPF);
        %             processData.Units.CAPEX(i) = co;
        %             processData.Units.Size(i) = si;
        %             CapitalCosts.separator = [CapitalCosts.separator, processData.Units.CAPEX(i)];

        % elseif strcmp(processData.Units.NAME(i), 'B-A')
        %
        %     stname='SA'; % Flash liquid
        %     LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1) / 3600; % [kg/s]
        %     LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        %     VesselPressure = 1; % processData.Units.PRESSURE{i}; % [bar]
        %     [co,si]= NPV_flash_2020(LFlow,LDensity,VesselPressure,COST.UPF);
        %     processData.Units.CAPEX(i) = co;
        %     processData.Units.Size(i) = si;
        %     CapitalCosts.separator = [CapitalCosts.separator, processData.Units.CAPEX(i)];

        if strcmp(processData.Units.NAME(i), 'S-101')

            %         elseif strcmp(processData.Units.NAME(i), 'S-101') % consider dryer

            stname='108'; %
            LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1) / 3600; % [kg/s]
            LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
            VesselPressure = 1; % processData.Units.PRESSURE{i}; % [bar]
            [co,si]= NPV_flash_2020(LFlow,LDensity,VesselPressure,COST.UPF);
            processData.Units.CAPEX(i) = co;
            processData.Units.Size(i) = si;
            CapitalCosts.separator = [CapitalCosts.separator, processData.Units.CAPEX(i)];

        elseif strcmp(processData.Units.NAME(i), 'S-102')

            stname='112'; % Flash liquid
            LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1) / 3600; % [kg/s]
            LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
            VesselPressure = 1; % processData.Units.PRESSURE{i}; % [bar]
            [co,si]= NPV_flash_2020(LFlow,LDensity,VesselPressure,COST.UPF);
            processData.Units.CAPEX(i) = co;
            processData.Units.Size(i) = si;
            CapitalCosts.separator = [CapitalCosts.separator, processData.Units.CAPEX(i)];

            %         elseif strcmp(processData.Units.NAME(i), 'S-103')
            %
            %             stname='141'; % Flash liquid
            %             LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1) / 3600; % [kg/s]
            %             LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
            %             VesselPressure = 1; % processData.Units.PRESSURE{i}; % [bar]
            %             [co,si]= NPV_flash_2020(LFlow,LDensity,VesselPressure,COST.UPF);
            %             processData.Units.CAPEX(i) = co;
            %             processData.Units.Size(i) = si;
            %             CapitalCosts.separator = [CapitalCosts.separator, processData.Units.CAPEX(i)];

        elseif strcmp(processData.Units.NAME(i), 'S-301') % PSA-H2 recycle

            stname='306';
            pwd = join(['Data\Streams\',stname,'\Output\MASSFLMX_GAS']);
            V_activecarbon = 1;
            m_PSA = h.Tree.FindNode(pwd).value;
            v_PSA = h.Tree.FindNode(pwd).value;

            P_Col = processData.Streams.PRESSURE(strcmp(processData.Streams.NAME(:), stname),1); % [bar]
            % P_Col = 30; % [bar]. Just a guess. Should be revised in the future.

            % V_activecarbon = calculate_adsorbent_volume(m_PSA, P_Col);
            [D_Col,H_Col,V_Col,cost]= NPV_PSA_2020(P_Col,V_activecarbon);
            CapitalCosts.PSA = [CapitalCosts.PSA cost];
            processData.Units.CAPEX(i) = cost;
            processData.Units.Size(i) = V_Col;
            % processData.Units.CAPEX(i) = 0 ;

        elseif strcmp(processData.Units.NAME(i), 'S-401') % PSA H2-Reference

            stname='419';
            pwd = join(['Data\Streams\',stname,'\Output\MASSFLMX_GAS']);
            V_activecarbon = 1;
            m_PSA = h.Tree.FindNode(pwd).value;
            v_PSA = h.Tree.FindNode(pwd).value;

            P_Col = processData.Streams.PRESSURE(strcmp(processData.Streams.NAME(:), stname),1); % [bar]
            % P_Col = 30; % [bar]. Just a guess. Should be revised in the future.
            % V_activecarbon = calculate_adsorbent_volume(m_PSA, P_Col);
            [D_Col,H_Col,V_Col,cost]= NPV_PSA_2020(P_Col,V_activecarbon);
            CapitalCosts.PSA = [CapitalCosts.PSA cost];
            processData.Units.CAPEX(i) = cost;
            processData.Units.Size(i) = V_Col;
            % processData.Units.CAPEX(i) = 0 ;

        end

        % elseif strcmp(processData.Units.NAME(i), 'S-401') % PSA H2-Reference
        %
        %     stname = '419';
        %     pwd_mass = join(['Data\Streams\',stname,'\Output\MASSFLMX_GAS']);
        %     pwd_vol = join(['Data\Streams\',stname,'\Output\VOLFLMX_GAS']);
        %
        %     m_PSA = h.Tree.FindNode(pwd_mass).value;
        %     v_PSA = h.Tree.FindNode(pwd_vol).value;
        %
        %     P_Col = determine_PSA_pressure(m_PSA);
        %     V_activecarbon = calculate_adsorbent_volume(m_PSA, P_Col);
        %     [D_Col, H_Col, V_Col, cost] = NPV_PSA_2020(P_Col, V_activecarbon);
        %
        %     CapitalCosts.PSA = [CapitalCosts.PSA cost];
        %     processData.Units.CAPEX(i) = cost;
        %     processData.Units.Size(i) = V_Col;
        % end



        %    % cost for dryer
        % if strcmp(processData.Units.TYPE(i), 'dryer') % dryer
        % elseif strcmp(processData.Units.NAME(i), 'S-101')
        %    % Units.Work is the evaporating moisture rate of the dryer.
        %    SFlow = h.Tree.FindNode(['\Data\Streams\108\Output\MASSFLMX\$TOTAL']).value; % [kg/hr]
        %    processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1) / 3600; % [kg/s]
        %    [co3, si3, mi3] = NPV_dryer_2020(SFlow,COST.UPF);
        %    processData.Units.CAPEX(unitCount) = co3;
        %    processData.Units.Size(unitCount) = si3;
        %    processData.Units.WORK(unitCount) = {mi3};
        %    CapitalCosts.dryer = [CapitalCosts.dryer, processData.Units.CAPEX(unitCount)];
        %
        % end

        % 2 phase flash
    elseif strcmp(processData.Units.TYPE(i), 'Flash2')
        stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'L(OUT)')); % Flash liquid
        LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1) / 3600; % [kg/s]
        LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        VesselPressure = processData.Units.PRESSURE{i}; % [bar]
        [co,si]= NPV_flash_2020(LFlow,LDensity,VesselPressure,COST.UPF);
        processData.Units.CAPEX(i) = co;
        processData.Units.Size(i) = si;
        CapitalCosts.flash2 = [CapitalCosts.flash2, processData.Units.CAPEX(i)];

        % 3 phase flash
    elseif strcmp(processData.Units.TYPE(i), 'Flash3')
        stname1 = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'L1(OUT)'));
        stname2 = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'L2(OUT)'));
        LFlow1 = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname1),1) / 3600; % [kg/s]
        LDensity1 = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname1),1) * 1000; % [kg/cum]
        LFlow2 = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname2),1) / 3600; % [kg/s]

        if LFlow2==0
            VesselPressure = processData.Units.PRESSURE{i}; % [bar]
            [co,si]= NPV_flash_2020(LFlow1,LDensity1,VesselPressure,COST.UPF);
            processData.Units.CAPEX(i) = co;
            processData.Units.Size(i) = si;
            CapitalCosts.flash3 = [CapitalCosts.flash3, processData.Units.CAPEX(i)];
        else
            LDensity2 = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname2),1) * 1000; % [kg/cum]
            VesselPressure = processData.Units.PRESSURE{i}; % [bar]
            [co,si]= NPV_flash3_2020(LFlow1,LDensity1,LFlow2,LDensity2,VesselPressure,COST.UPF);
            processData.Units.CAPEX(i) = co;
            processData.Units.Size(i) = si;
            CapitalCosts.flash3 = [CapitalCosts.flash3, processData.Units.CAPEX(i)];
        end

        % Cost for Stoic Reactor
    elseif strcmp(processData.Units.TYPE(i), 'RStoic')
        stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)')); % Flash liquid
        stnameIN = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{1,1}(2,:), 'F(IN)')); % Flash liquid
        try
            vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stnameIN)) * 60; % [L/h]
        catch
            stn = stnameIN{1};
            % vFLow = processData.Streams.VOLFLOW{strcmp(processData.Streams.NAME(:), stn)} * 60; % [L/h]
            vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stn)) * 60;  % [L/h]
        end

        LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1)/ 3600; % [kg/s]
        LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        VesselPressure = processData.Units.PRESSURE{i}; % [bar]

        ReactorVol = LFlow * 60 * 5 * 2;
        UPF_temp = 1; % for commercialize scale, it can be considered as 1

        [co,si]= NPV_CSTR_2020(LFlow,LDensity,VesselPressure,UPF_temp,ReactorVol);

        processData.Units.CAPEX(i) = co;
        processData.Units.Size(i) = si;
        CapitalCosts.reactor = [CapitalCosts.reactor, processData.Units.CAPEX(i)];

        % elseif strcmp(processData.Units.TYPE(i), 'RStoic') && ~strcmp(processData.Units.NAME{i}, 'PLASMA-P')
        % stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)')); % Flash liquid
        % stnameIN = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{1,1}(2,:), 'F(IN)')); % Flash liquid
        % try
        %     vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stnameIN)) * 60; % [L/h]
        % catch
        %     stn = stnameIN{1};
        %     % vFLow = processData.Streams.VOLFLOW{strcmp(processData.Streams.NAME(:), stn)} * 60; % [L/h]
        %     vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stn)) * 60;  % [L/h]
        % end
        %
        % % if strcmp(processData.Units.NAMW(i), 'ETP1')
        % % LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1)/ 3600;
        % % else
        % LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1)/ 3600; % [kg/s]
        % % end
        % % LFlow = sum([LFlow(:,2)]);
        % LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        % VesselPressure = processData.Units.PRESSURE{i}; % [bar]
        %
        % ReactorVol = LFlow * 60 * 5 * 2;
        %
        % % UPF correction for SOL1/SOL2 Reactors
        % if strcmp(processData.Units.NAME(i), 'SOL1-REA')
        %     UPF_temp = 3;
        % elseif strcmp(processData.Units.NAME(i), 'SOL2-REA')
        %     UPF_temp = 3;
        % elseif strcmp(processData.Units.NAME(i), 'ELEC-REA')
        %     UPF_temp = 0;
        % else
        %     UPF_temp = COST.UPF;
        % end
        %
        % [co,si]= NPV_CSTR_2020(LFlow,LDensity,VesselPressure,UPF_temp,ReactorVol);
        %
        % processData.Units.CAPEX(i) = co;
        % processData.Units.Size(i) = si;
        % CapitalCosts.reactor = [CapitalCosts.reactor, processData.Units.CAPEX(i)];

    elseif strcmp(processData.Units.TYPE(i), 'RYield')
        stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)')); % Flash liquid
        stnameIN = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{1,1}(2,:), 'F(IN)')); % Flash liquid
        try
            vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stnameIN)) * 60; % [L/h]
        catch
            stn = stnameIN{1};
            % vFLow = processData.Streams.VOLFLOW{strcmp(processData.Streams.NAME(:), stn)} * 60; % [L/h]
            vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stn)) * 60;  % [L/h]
        end

        LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1)/ 3600; % [kg/s]
        LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        VesselPressure = processData.Units.PRESSURE{i}; % [bar]

        ReactorVol = LFlow * 60 * 5 * 2;
        UPF_temp = 1; % for commercialize scale, it can be considered as 1

        [co,si]= NPV_CSTR_2020(LFlow,LDensity,VesselPressure,UPF_temp,ReactorVol);

        processData.Units.CAPEX(i) = co;
        processData.Units.Size(i) = si;
        CapitalCosts.reactor = [CapitalCosts.reactor, processData.Units.CAPEX(i)];

    elseif strcmp(processData.Units.TYPE(i), 'RGibbs')
        stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)')); % Flash liquid
        stnameIN = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{1,1}(2,:), 'F(IN)')); % Flash liquid
        try
            vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stnameIN)) * 60; % [L/h]
        catch
            stn = stnameIN{1};
            % vFLow = processData.Streams.VOLFLOW{strcmp(processData.Streams.NAME(:), stn)} * 60; % [L/h]
            vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stn)) * 60;  % [L/h]
        end

        LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1)/ 3600; % [kg/s]
        LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        VesselPressure = processData.Units.PRESSURE{i}; % [bar]

        ReactorVol = LFlow * 60 * 5 * 2;
        UPF_temp = 1; % for commercialize scale, it can be considered as 1

        [co,si]= NPV_CSTR_2020(LFlow,LDensity,VesselPressure,UPF_temp,ReactorVol);

        processData.Units.CAPEX(i) = co;
        processData.Units.Size(i) = si;
        CapitalCosts.reactor = [CapitalCosts.reactor, processData.Units.CAPEX(i)];

        % elseif strcmp(processData.Units.TYPE(i), 'RYield') && ~strcmp(processData.Units.NAME{i}, 'PLASMA-R')
        %     stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)')); % Flash liquid stream
        %     stnameIN = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'F(IN)')); % Flash liquid stream
        %     try
        %         vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stnameIN)) * 60; % [L/h]
        %     catch
        %         stn = stnameIN{1};
        %         % vFLow = processData.Streams.VOLFLOW{strcmp(processData.Streams.NAME(:), stn)} * 60; % [L/h]
        %         vFlow = processData.Streams.VOLFLOW(strcmp(processData.Streams.NAME(:), stn)) * 60;  % [L/h]
        %     end
        %
        %     % if strcmp(processData.Units.NAMW(i), 'ETP1')
        %     % LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1)/ 3600;
        %     % else
        %     LFlow = processData.Streams.MASSFLOW(strcmp(processData.Streams.NAME(:), stname),1)/ 3600; % [kg/s]
        %     % end
        %     % LFlow = sum([LFlow(:,2)]);
        %     LDensity = processData.Streams.DENSITY(strcmp(processData.Streams.NAME(:), stname),1) * 1000; % [kg/cum]
        %     VesselPressure = processData.Units.PRESSURE{i}; % [bar]
        %
        %     ReactorVol = LFlow * 60 * 5 * 2;
        %
        %     % UPF correction for SOL1/SOL2 Reactors
        %     if strcmp(processData.Units.NAME(i), 'SOL1-REA')
        %         UPF_temp = 3;
        %     elseif strcmp(processData.Units.NAME(i), 'SOL2-REA')
        %         UPF_temp = 3;
        %     elseif strcmp(processData.Units.NAME(i), 'ELEC-REA')
        %         UPF_temp = 0;
        %     else
        %         UPF_temp = COST.UPF;
        %     end
        %
        %     [co,si]= NPV_CSTR_2020(LFlow,LDensity,VesselPressure,UPF_temp,ReactorVol);
        %
        %     processData.Units.CAPEX(i) = co;
        %     processData.Units.Size(i) = si;
        %     CapitalCosts.reactor = [CapitalCosts.reactor, processData.Units.CAPEX(i)];

        % Distillation Column
    elseif strcmp(processData.Units.TYPE(i), 'RadFrac')
        % P_col = input.pressure/10^5; % [Pa to bar]
        S_col = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Input\NSTAGE']).value; % number of tray
        P_col = cell2mat(processData.Units.PRESSURE(i)); % [bar]
        % oiRebler
        if strcmp(processData.Units.TYPE{i},'Distl')
            feedtemp = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\FEED_TRAY_T']).value;
            stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'D(OUT)'));
            COND_index = 'COND_TYPE';
        elseif strcmp(processData.Units.TYPE{i},'RadFrac')

            try
                try
                    feedtemp = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\B_TEMP\',num2str(floor(S_col/2))]).value +273.15; % [K]
                catch

                    feedtemp = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\B_TEMP\2']).value +273.15; % [K]

                end
            catch
                feedtemp = 200+273.15;
            end
            stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'VD(OUT)'));
            COND_index = 'CONDENSER'
        end
        bottomtemp = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\BOTTOM_TEMP']).value +273.15; %[K]
        heat_duty = processData.Units.REBDUTY{i}; % [cal/s]
        % steam_temp = 130 + 273.15;
        % cost_reboiler = heatX(steam_temp, steam_temp, feedtemp,bottomtemp,P_col,heat_duty.COST.UPF);

        if bottomtemp < 147.778 + 273.15
            T1 = 147.778+273.15 - feedtemp;
            T2 = 147.778+273.15 - bottomtemp;
        elseif bottomtemp < 185.556 + 273.15
            T1 = 185.556+273.15 - feedtemp;
            T2 = 185.556+273.15 - bottomtemp;
        else
            T1 = 500+273.15 - feedtemp;
            T2 = 500+273.15 - bottomtemp;

        end
        cost_reboiler = NPV_heater_2020(T1,T2,P_col,heat_duty,COST.UPF);

        % Condenser
        toptemp = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\TOP_TEMP']).value + 273.15; % [K]
        heat_duty = processData.Units.CONDDUTY{i}; % [cal/s]

        if toptemp > 30+273.15
            refri_temp = 30+273.15;
        elseif toptemp > 4.444+ 273.15
            refri_temp = 4.444+ 273.15;
        elseif toptemp > -12.2222+273.15
            refri_temp = -12.2222+273.15;
        elseif toptemp > -62.222+273.15
            refri_temp = -62.2222+273.15;
        else
            refri_temp = -101.111+273.15;
        end

        cost_condenser = heatX(feedtemp,toptemp,refri_temp, refri_temp,P_col,heat_duty,COST.UPF);

        Sigma = 72; % [dyne/cm] 72 for water

        clear MSF_v MOF_v Rho_v_MOLE

        try
            for j = 1 : S_col
                MSF_v(j) = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\VAP_FLOW_MS\',num2str(j)]).value; % kg/hr
                Rho_v_MOLE(j) = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\RHO_GAS\',num2str(j)]).value; % kmol/hr
                MOF_v(j) = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\VAP_FLOW\',num2str(j)]).value % kmol/hr
            end

            [Rho_v, ind_v] = max(MSF_v./(MOF_v ./ Rho_v_MOLE));
            Rho_v = Rho_v * 10^9;

            try
                MSF_l = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i},'\Output\LIQ_FLOW_MS\',num2str)]).value;
            catch

                MSF_l = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\LIQ_FLOW_MS\',num2str(ind_v)]).value;

            end

            Rho_l_MOLE =  h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\RHO_LIQ\',num2str(ind_v)]).value;
            MOF_l = h.Tree.FindNode(['\Data\Blocks\',char(processData.Units.NAME{i}),'\Output\LIQ_FLOW\',num2str(ind_v)]).value;

            Rho_l = MSF_l./(MOF_l ./ Rho_l_MOLE) *10^9;

            m_l = MSF_l / 3600; % [kg/s]
            m_v = MSF_v(ind_v) / 3600; % [kg/s]

            [D_Col,H_Col,V_Col,cost_column] = NPV_distillation_2020(P_col,S_col, Sigma,Rho_l,Rho_v,m_v,m_l);

            % cost
            try
                cost = cost_column + cost_reboiler + cost_condenser;
            catch
                disp('no data for detailed cost')
                cost = 10*1e8;
                V_Col = 990;
                D_Col = 990;
                H_Col = 990;
            end

        catch
            cost = 1e8;
            V_Col = 999;
            D_Col = 999;
            H_Col = 999;

        end

        CapitalCosts.distillation = [CapitalCosts.distillation cost];
        processData.Units.Size(i) = V_Col;
        processData.Units.Diameter(i) = D_Col;
        processData.Units.Height(i) = H_Col;
        processData.Units.CAPEX(i) = cost;

    end

    % Compressor ($)

    if strcmp(processData.Units.TYPE{i},'Compr')
        Work = processData.Units.WORK{i}*1000; % [W]
        [cost] = NPV_compressor_2020(Work,COST.UPF);
        % [cost] = real(cost);
        CapitalCosts.compressor = [CapitalCosts.compressor cost];
        processData.Units.CAPEX(i) = cost;
    end

    % Pump ($)

    if strcmp(processData.Units.TYPE{i},'Pump')
        Work = processData.Units.WORK{i}*1000; % [W]
        [cost] = NPV_compressor_2020(Work,COST.UPF);
        % [cost] = real(cost);
        CapitalCosts.pump = [CapitalCosts.pump cost];
        processData.Units.CAPEX(i) = cost;
    end

    % % Expander ($)
    %
    % if strcmp(processData.Units.TYPE{i},'Expander')
    %     Work = processData.Units.WORK{i}*1000; % [W]
    %     [cost] = NPV_expander_2020(Work,COST.UPF);
    %     % [cost] = real(cost);
    %     CapitalCosts.Expander = [CapitalCosts.pump cost];
    %     processData.Units.CAPEX(i) = cost;
    % end

    %% heater

    % if strcmp(processData.Units.NAME{i}, 'COOLER-3')
    %
    %     stname ='S23';
    %     T1 = processData.Streams.TEMPERATURE(strcmp(processData.Stream.NAME(:), stname),1)+273.15; % [K]
    %
    %     stname = processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
    %     T2 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1)+273.15; % [K]
    %     P = processData.Units.PRESSURE{i}; % [bar]
    %     heat_duty = processData.Units.HEATDUDY{i}; % [cal/s]
    %
    %     if T2>T1
    %         if T1> 237.778
    %                T1 = 260+273.15 -T1;
    %                T2 = 260+273.15 - T2;
    %            elseif T1> 185.556
    %                T1 = 237.778+273.15 - T1;
    %                T2 = 237.778+273.15 - T2;
    %            elseif T1> 147.778
    %                T1 = 185.556+273.15 - T1;
    %                T2 = 185.556+273.15 - T2;
    %            else
    %                T1 = 147.778+273.15 - T1;
    %                T2 = 147.778+273.15 - T2;
    %         end
    %     else
    %
    %         TempT = T1;
    %         T1 = T2;
    %         T2 = TempT;
    %
    %         if T1>30 + 273.15
    %             T1 = T1 - 30 - 273.15;
    %             T2 = T2 - 30 - 273.15;
    %         elseif T1>4+273.15
    %             T1 = T1 - 4 - 273.15;
    %             T2 = T2 - 4 - 273.15;
    %         else
    %             T1 = T1 +50 - 273.15;
    %             T2 = T2 +50 - 273.15;
    %         end
    %
    %     end
    %     [cost, Area] = NPV_heater_2020(T1,T2,P,heat_duty,COST.UPF);
    %     CapitalCosts.heater = [CapitalCosts.heater cost];
    %     processData.Units.CAPEX(i) = cost;
    %     processData.Units.Size(i) = Area;
    % end

    if strcmp(processData.Units.TYPE{i},'Heater') %&& ~strcmp(processData.Units.NAME{i}, 'COOLER-3;) && ~strcmp(processData.Units.NAME{i},'B25')
        stname1=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'HS(IN)'));
        stname2=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'HS(OUT)'));

        if isempty(stname1) && isempty(stname2)
            stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'F(IN)'));
            T1 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]

            stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
            T2 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1) + 273.15; % [K]
            P = processData.Units.PRESSURE{i}; % [bar]
            heat_duty = processData.Units.HEATDUTY{i}; % [cal/s]

            if T2>T1
                if T2>273.778+273.15
                    T1 = abs(265+273.15 - T1);
                    T2 = abs(265+273.15 - T2);
                elseif T2> 185.556+273.15
                    T1 = abs(237.778+273.15 - T1);
                    T2 = abs(237.778+273.15 - T2);
                elseif T2> 147.778+273.15
                    T1 = abs(185.556+273.15 - T1);
                    T2 = abs(185.556+273.15 - T2);
                else
                    T1 = abs(147.778+273.15 - T1);
                    T2 = abs(147.778+273.15 - T2);
                end
            else
                TempT = T1;
                T1 = T2;
                T2 = TempT;

                if T1>30 + 273.15
                    T1 = abs(T1 - 30 - 273.15);
                    T2 = abs(T2 - 30 - 273.15);
                elseif T1>4+273.15
                    T1 = abs(T1 - 4 - 273.15);
                    T2 = abs(T2 - 4 - 273.15);
                else
                    T1 = abs(T1 +50 - 273.15);
                    T2 = abs(T2 +50 - 273.15);
                end
            end

            [cost,Area] = NPV_heater_2020(T1,T2,P,heat_duty,COST.UPF);
            CapitalCosts.heater = [CapitalCosts.heater cost];
            processData.Units.CAPEX(i) = cost;
            processData.Units.Size(i) = Area;

        else
            clear T1 T2 P heat_duty
            stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'F(IN)'));
            T1 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1)+273.15; % [K]

            stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
            T2 = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1)+273.15; % [K]

            if T1 >= T2 % prevent double calculation
                cost = 0;
                Area = 0;

                processData.Units.CAPEX(i) = cost;
                processData.Units.Size(i) = Area;
            else
                P = processData.Units.PRESSURE{i}; % [bar]
                heat_duty = processData.Units.HEATDUTY{i}; % [cal/s]

                if ~isempty(stname1)
                    if processData.Streams.DUTY{strcmp(processData.Streams.NAME(:), stname1),1} == 0

                        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'F(IN)'));
                        T1 = processData.Streams.TEMPERATURE{strcmp(processData.Streams.NAME(:), stname),1}+273.15; % [K]

                        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
                        T2 = processData.Streams.TEMPERATURE{strcmp(processData.Streams.NAME(:), stname),1}+273.15; % [K]
                        P = processData.Units.PRESSURE{i}; % [bar]
                        heat_duty = processData.Units.HEATDUTY{i}; % [cal/s]

                        if T2>T1
                            if T1>237.778+273.15
                                T1 = 260+273.15 - T1;
                                T2 = 260+273.15 - T2;
                            elseif T1> 185.556+273.15
                                T1 = 237.778+273.15 - T1;
                                T2 = 237.778+273.15 - T2;
                            elseif T1> 147.778+273.15
                                T1 = 185.556+273.15 - T1;
                                T2 = 185.556+273.15 - T2;
                            else
                                T1 = 147.778+273.15 - T1;
                                T2 = 147.778+273.15 - T2;
                            end
                        else
                            TempT = T1;
                            T1 = T2;
                            T2 = TempT;

                            if T1>30 + 273.15
                                T1 = T1 - 30 - 273.15;
                                T2 = T2 - 30 - 273.15;
                            elseif T1>4+273.15
                                T1 = T1 - 4 - 273.15;
                                T2 = T2 - 4 - 273.15;
                            else
                                T1 = T1 +50 - 273.15;
                                T2 = T2 +50 - 273.15;
                            end
                        end

                        [cost,Area] = NPV_heater_2020(T1,T2,P,heat_duty,COST.UPF);
                        CapitalCosts.heater = [CapitalCosts.heater cost];
                        processData.Units.CAPEX(i) = cost;
                        processData.Units.Size(i) = Area;
                    else
                        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'HS(IN)'));
                        THin = processData.Streams.T_begin{strcmp(processData.Streams.NAME(:), stname),1}+273.15; % [K]
                        THout = processData.Streams.T_end{strcmp(processData.Streams.NAME(:), stname),1}+273.15; % [K]

                        TCin = T1; % [K]
                        TCout = T2; % [K]

                        [cost, Area] = heatX(THin,THout,TCin, TCout,P,heat_duty,COST.UPF);
                        CapitalCosts.heatX = [CapitalCosts.heatX cost];
                        processData.Units.CAPEX(i) = cost;
                        processData.Units.Size(i) = Area;
                    end

                elseif ~isempty(stname2)
                    if processData.Streams.DUTY{strcmp(processData.Streams.NAME(:), stname2),1} == 0
                        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'F(IN)'));
                        T1 = processData.Streams.TEMPERATURE{strcmp(processData.Streams.NAME(:), stname),1}+273.15; % [K]
                        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'P(OUT)'));
                        T2 = processData.Streams.TEMPERATURE{strcmp(processData.Streams.NAME(:), stname),1}+273.15; % [K]
                        P = processData.Units.PRESSURE{i}; % [bar]
                        heat_duty = processData.Units.HEATDUTY{i}; % [cal/s]

                        if T2>T1
                            if T1>237.778+273.15
                                T1 = 260+273.15 - T1;
                                T2 = 260+273.15 - T2;
                            elseif T1> 185.556+273.15
                                T1 = 237.778+273.15 - T1;
                                T2 = 237.778+273.15 - T2;
                            elseif T1> 147.778+273.15
                                T1 = 185.556+273.15 - T1;
                                T2 = 185.556+273.15 - T2;
                            else
                                T1 = 147.778+273.15 - T1;
                                T2 = 147.778+273.15 - T2;
                            end
                        else
                            TempT = T1;
                            T1 = T2;
                            T2 = TempT;

                            if T1>30 + 273.15
                                T1 = T1 - 30 - 273.15;
                                T2 = T2 - 30 - 273.15;
                            elseif T1>4+273.15
                                T1 = T1 - 4 - 273.15;
                                T2 = T2 - 4 - 273.15;
                            else
                                T1 = T1 +50 - 273.15;
                                T2 = T2 +50 - 273.15;
                            end
                        end

                        [cost, Area] = NPV_heater_2020(T1,T2,P,heat_duty,COST.UPF);
                        CapitalCosts.heater = [CapitalCosts.heater cost];
                        processData.Units.CAPEX(i) = cost;
                        processData.Units.Size(i) = Area;
                    end
                end
            end
        end
    end

    if strcmp(processData.Units.TYPE{i},'HeatX')
        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'H(IN)'));
        THin = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1)+273.15; % [K]

        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'H(OUT)'));
        THout = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1)+273.15; % [K]

        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'C(IN)'));
        TCin = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1)+273.15; % [K]

        stname=processData.Units.CONNECTION{i,1}(1,strcmp(processData.Units.CONNECTION{i,1}(2,:), 'C(OUT)'));
        TCout = processData.Streams.TEMPERATURE(strcmp(processData.Streams.NAME(:), stname),1)+273.15; % [K]

        P = processData.Units.PRESSURE{i}; % [bar]
        heat_duty = processData.Units.HEATDUTY{i}; % [cal/s]

        [cost, Area] = heatX(THin,THout,TCin,TCout,P,heat_duty,COST.UPF);
        CapitalCosts.heatX = [CapitalCosts.heatX cost];
        processData.Units.CAPEX(i) = cost;
        processData.Units.Size(i) = Area;

    end

    % C_base = 11.2e+06;            % Base CAPEX [$]
    % CO2_base = 12440;             % Base flow [kg/h]

    % 250607 ---> https://doi.org/10.1016/j.ifacol.2022.07.458
    % % C_TBM = 1.9606e+07 / 1.68;            % Base C_TBM [$]
    % % C_base = C_TBM;                       % Base CAPEX [$]
    % CO2_base = 196000                       % [t/y]
    % % CO2_base = 24500;                     % Base flow [kg/h]

    % % C_cap = C_base * (processData.Output.CO2_CO2CAPTURE.Massflow / CO2_base)^0.6;
    % % CapitalCosts.Capture = C_cap;
    % % processData.CapitalCost = CapitalCosts;

    % T_op = 8000;                  % Operating time [h/year]
    % BaseIndex_2013 = 567.3;  % CEPCI
    % BaseIndex_2022 = 816; % CEPCI
    % Carbon_capture_without_sulfur_removal_2013 = 44; % [$/ton CO2] https://pubs.acs.org/doi/full/10.1021/ie402185h
    % EscalationFactor = BaseIndex_2022 / BaseIndex_2013;
    % Carbon_capture_without_sulfur_removal = Carbon_capture_without_sulfur_removal_2013 * EscalationFactor;
    % Portion_carbon_capture_CAPEX = 0.514;
    % C_TBM_unit = Carbon_capture_without_sulfur_removal * Portion_carbon_capture_CAPEX; % [$/tonne]
    % 
    % Plant_capacity_CC = 500; % [MW]
    % CO2_emission_intensity = 0.808; % [CO2/MWh]
    % CO2_base = Plant_capacity_CC * CO2_emission_intensity * 1000; % kg/h
    % 
    % C_annual = C_TBM_unit / 1000 * CO2_base * T_op;  % $/yr
    % % return on investment, r, is taken here as 0.2 (20%)
    % C_TCI_base = C_annual / 0.2;   % $
    % C_TBM_base = (C_TCI_base / 1.68);  % $
    % C_cap = C_TBM_base * (processData.Output.CO2_CO2CAPTURE.Massflow / CO2_base)^0.6;
    % CapitalCosts.Capture = C_cap;

    % Catalyst CAPEX
    C_catalyst = 59707.35; %$
    Catalyst_CAPEX = C_catalyst;
    CapitalCosts.Catalyst = Catalyst_CAPEX;

    processData.CapitalCost = CapitalCosts;
end