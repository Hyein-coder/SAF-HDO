% Sizing and Costing of Heat Exchangers
% Free Convection Gas - Free Convection Gas : U = 1 - 2 W/m2K  (typical window, room to outside air through glass)
% Free Convection Gas - Forced liquid (flowing) water : U = 5 - 15 W/m2K  (typical radiator central heating)
% Free Convection Gas - Condensing Vapor Water : U = 5 - 20 W/m2K  (typical steam radiators)
% Forced Convection (flowing) Gas - Free Convection Gas : U = 3 - 10 W/m2K (superheaters)
% Forced Convection (flowing) Gas - Forced Convection Gas : U = 10 - 30 W/m2K (heat exchanger gases)
% Forced Convection (flowing) Gas - Forced liquid (flowing) water  : U = 10 - 50 W/m2K (gas coolers)
% Forced Convection (flowing) Gas - Condensing Vapor Water  : U = 10 - 50 W/m2K (air heaters)
% Liquid Free Convection - Forced Convection Gas : U = 10 - 50 W/m2K (gas boiler)
% Liquid Free Convection - Free Convection Liquid : U = 25 - 500 W/m2K (oil bath for heating)
% Liquid Free Convection - Forced Liquid flowing (Water) : U = 50 - 100 W/m2K (heating coil in vessel water, water without steering), 500 - 2000 W/m2K (heating coil in vessel water, water with steering)
% Liquid Free Convection - Condensing vapor water : U = 300 - 1000 W/m2K (steam jackets around vessels with stirrers, water), 150 - 500 W/m2K (other liquids)
% Forced liquid (flowing) water - Free Convection Gas : U = 10 - 40 W/m2K (combustion chamber + radiation)
% Forced liquid (flowing) water - Free Convection Liquid : U = 500 - 1500 W/m2K (cooling coil - stirred)
% Forced liquid (flowing) water - Forced liquid (flowing) water : U = 900 - 2500 W/m2K (heat exchanger water/water)
% Forced liquid (flowing) water - Condensing vapor water : U = 1000 - 4000 W/m2K (condensers steam water)
% Boiling liquid water - Free Convection Gas : U = 10 - 40 W/m2K (steam boiler + radiation)
% Boiling liquid water - Forced Liquid flowing (Water) : U = 300 - 1000 W/m2K (evaporation of refrigerators or brine coolers)
% Boiling liquid water - Condensing vapor water : U = 1500 - 6000 W/m2K (evaporators steam/water)

function [cost, Area] = heatX(THin,THout,TCin, TCout,P,heat_duty,UPF)

    % Conversion factors
%     U = 0.0203019012*4.1868/0.0001;
     U = 850;
%    U = 800;
    
    T1 = THin - TCout;
    T2 = THout - TCin;
    
%     U = 3975;
    P_psigtokpa = 100/14.5; % 1 psi
    P_bartokpa = 100;
    P = P*P_bartokpa;
    A_ft2tom2 = 1/(3.2808^2);
    
    heat_duty = heat_duty * UPF;
    Q = heat_duty*4.1868; % cal in J umrechnen
    
    %Sizing
    Tln = (T1-T2)/log(T1/T2); %Mean temperature
    Area = abs(Q)/(U*Tln); %required heat transfer area
    
    %Cost calculation
    C0 = 5000;          % $
    S0 = 400*A_ft2tom2; % m^2
    alpha = 0.65;
    Amax = 10000*A_ft2tom2; % maximum Area
    
    %Calculating whether Area is within Costing Range, otherwise dividing unit
    %into several units
    if Area <= Amax
        Amount = 1;
    else
        Amount = ceil(Area/Amax);
        Area = Area/Amount;
    end
    
    BC = C0*(Area/S0)^alpha; %Base Cost
    
    Fm = 1; %Guthrie design type: floating head
    % Fd = 1; %Guthrie Material: Carbon steel
    % Fd = 2.25; %Material type ; 251110 for SS316 clad
    Fd = 2;  % 251113 for SS from Guthrie

    Fp = 0; %Guthrie Pressure Factor
    
    if P <= (500*P_psigtokpa)
        Fp = 0;
    elseif (500*P_psigtokpa) <P && P<= (1000*P_psigtokpa)
        Fp = 0.1;
    elseif (1000*P_psigtokpa) <P && P<= (1500*P_psigtokpa)
        Fp = 0.15;
    elseif (1500*P_psigtokpa) <P && P<= (2000*P_psigtokpa)
        Fp = 0.25;
    elseif (2000*P_psigtokpa) <P && P<= (2500*P_psigtokpa)
        Fp = 0.4;
    elseif (2500*P_psigtokpa) <P && P<= (3000*P_psigtokpa)
        Fp = 0.6;
    end
    
    MPF = Fm*(Fp+Fd); %Guthrie Material and Pressure Factor
    
    if BC<=2e5
        MF = 3.29;
    elseif 2e5<BC && BC<=4e5
        MF = 3.18;
    elseif 4e5<BC && BC<=6e5
        MF = 3.14;
    elseif 6e5<BC && BC<=8e5
        MF = 3.12;
    elseif 8e5<BC && BC<=10e5
        MF = 3.09;
    else
        MF = 3.09 ;
%         fprintf('BC is out of order in the heater cost function') ;
    end
    
    % UF = 576.1/113; %conversion of cost from time of Guthries book to present
    UF = 816/113; % 2022/Guthrie
    
    BMC = UF*BC*(MF+MPF-1); %updated bare module cost
cost = BMC * Amount;
    


end
