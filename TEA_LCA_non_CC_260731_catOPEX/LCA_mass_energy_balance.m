% In-Out Mass & Energy Balance 

me_balance = struct();

me_balance.Case = string(list_simulations{idxSim});

nhv_saf = 48.168;
me_balance.SAF_absolute = processData.Output.SAF.Massflow * nhv_saf;  % MJ/hr
me_balance.SAF = 1 / nhv_saf;
me_balance.biomass = processData.Input.WF.Massflow / me_balance.SAF_absolute; % kg/MJ
me_balance.NG = processData.Input.NG.Massflow / me_balance.SAF_absolute; % kg/MJ

me_balance.electricity = sum(processData.LCA.elec) / me_balance.SAF_absolute; % kWh/MJ
me_balance.steam = sum(processData.LCA.steam) / nhv_saf;  % MJ/MJ
me_balance.refrigerant = sum(processData.LCA.ref) / nhv_saf;  % MJ/MJ

me_balance.waste_water = processData.Output.WASTE_W.Massflow / me_balance.SAF_absolute; % kg/MJ
me_balance.gasoline = processData.Output.Gasoline.Massflow / me_balance.SAF_absolute; % kg/MJ

