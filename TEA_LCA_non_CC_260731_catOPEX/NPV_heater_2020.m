% Sizing and Costing of Heat Exchangers

function [cost, Area] = NPV_heater_2020(T1,T2,P,heat_duty,UPF)
 if (abs(T1-T2) > 0.000001)
    % Conversion factors
    %U = 0.0203019012*4.1868/0.0001;
    % https://www.engineersedge.com/thermodynamics/overall_heat_transfer-table.htm
    % 에서 U 값을 확인해보기..
     U = 850;
%U = 800;
    P_psigtokpa = 100/14.5; %원래 Pa로 들어오고 kPa로 바꾸기만 %100/14.5;
        P_bartokpa = 100;
    P = P*P_bartokpa;
    A_ft2tom2 = 1/(3.2808^2);
    
    if heat_duty < 0 
        heat_duty = abs(heat_duty);
    end
    heat_duty = heat_duty * UPF;
    Q = heat_duty*4.1868; % input is J *4.1868; % cal in J umrechnen
   
    %Sizing
    Tln = (T1-T2)/log(T1/T2); %Mean temperature
%     Tln = abs(T2-T1)/log(T2/T1); %Mean temperature
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
    Fd = 2; %Material type ;  251113 for SS from Guthrie

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
    
    % UF = 596.2/113; %conversion of cost from time of Guthries book to present
    UF = 816/113; % 2022 / Guthrie
    
    BMC = UF*BC*(MF+MPF-1); %updated bare module cost
cost = BMC * Amount;
    
else
    cost = 0.0 ;
    Area = 0;
end
end
