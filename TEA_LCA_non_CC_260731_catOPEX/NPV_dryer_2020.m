function [cost,V,moist] = NPV_dryer_2020(SFlow,UPF)
if (abs(SFlow) > 0.000001)
    
    % Function which calculates capital dryer cost
    % Input in SI
    %
    % INPUT
    %------------------------------------------------
    % Name           | Description                                | Size |
    % SFlow          | Solid mass flow of stream                  |[1x1] |
    % UPF            | Upscale factor                             |[1x1] |
    %------------------------------------------------
    %
    % OUTPUT
    %------------------------------------------------
    % Name  | Description                                         | Size |
    % cost  | upscaled capital cost of flash in dollar           |[1x1] |
    % V     | Peripheral area of the dryer                         |[1x1]|
    %------------------------------------------------
    
    %assumptions: Horizontal fabrication, L/D=4
    
%     VesselPressure = VesselPressure/10^5; %Pa to bar
%Cost calculation
    SFlow = UPF * SFlow; % No unit transition. Just UPF.
    evap_rate = 0.908 ; % [kg/hr/ft3]. Assume 2lb/hr/ft3 rate, from p.546. Calculations in CostAnalysis.xlsx.
    moisture_rate = SFlow*0.01*(10-1) ; % Assume 10% moisture to 1%, from p.546. Needs update.
    dry_vol = moisture_rate ./ evap_rate; %[ft3]
    len_to_dia = 6; % Assume lenght-to-dia ratio of 6, from p.546.
    rad_drum = nthroot(dry_vol/12/pi,3); % [ft]
    peri_area = 2*pi*rad_drum*(2*len_to_dia*rad_drum); % [ft2]
    
    UF = 596.2/394; % CEPCI for 2020.
    FBM = 3.291;    % bare-module factor, from p.490
    FOB = exp(10.158+0.1003*log(peri_area)+0.04303*(log(peri_area)^2)) ;    % FOB of dryer, stainless steel
    BMC = UF*FBM*FOB ;      % Need to check conversion from FOB to BMC 
    cost = BMC ; 
    V = peri_area;
    moist = moisture_rate ; 
    
else
    cost = 0.0 ;
    V = 0;
    moist = 0 ;
end

end