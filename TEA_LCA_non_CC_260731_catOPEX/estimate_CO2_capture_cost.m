function [Capex_CO2, Opex_CO2] = estimate_CO2_capture_cost(CO2_kgph)
    C_base = 5.250820292656341e+07;              % Base CAPEX [$]
    CO2_base = 12440;             % Base flow [kg/h]
    G_re = 4;                     % Energy [GJ/ton]
    C_steam = 0.0066;             % Steam cost [$/kg]
    T_op = 8000;                  % Operating time [h/year]
    MJ_per_GJ = 1000;
    steam_energy_per_kg = 2.326;  % [MJ/kg]

    %% CAPEX
    Capex_CO2 = C_base * (CO2_kgph / CO2_base)^0.6;

    %% OPEX
    CO2_tph = CO2_kgph / 1000;
    steam_MJph = CO2_tph * G_re * MJ_per_GJ;
    steam_kgph = steam_MJph / steam_energy_per_kg;
    Opex_CO2 = steam_kgph * C_steam * T_op;
end
