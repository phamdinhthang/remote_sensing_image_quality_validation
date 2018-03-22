%This function is used to extract the low frequency part of the DS from the
%CPF file which is computed by Astrium to include the fix for the
%low-frequency stray light effect


%1. Read DS from the Astrium reference CPF files (which include the
%straylight correction.
MyXMLToolsDir = 'D:\VNREDSat-1_NAOMI_125_calibration\Matlab_Code\xml_io_tools_2010_11_05';
addpath(genpath(MyXMLToolsDir))
MyCPFPath = 'C:\Users\HomeAdmin\Desktop\Sep_2015_Calib\Astrium_ref_CPF_straylight_correction';
MyCPFName = 'VNREDSAT_1_20141226_093000_20141227_000001.cpf';
MyCPF = fullfile(MyCPFPath,MyCPFName);
MyCPFTree = xml_read(MyCPF);
gen_object_display(MyCPFTree);
DS_PAN_CPF_ref=str2num(MyCPFTree.RadiometricParameters.PAN.G1.DarkCurrents);
DS_B1_CPF_ref=str2num(MyCPFTree.RadiometricParameters.B1.G1.DarkCurrents);
DS_B2_CPF_ref=str2num(MyCPFTree.RadiometricParameters.B2.G1.DarkCurrents);
DS_B3_CPF_ref=str2num(MyCPFTree.RadiometricParameters.B3.G1.DarkCurrents);
DS_B4_CPF_ref=str2num(MyCPFTree.RadiometricParameters.B4.G1.DarkCurrents);



%2. Read DS from the newly computed CPF with new DS
MyCPFPath = 'C:\Users\HomeAdmin\Desktop\Sep_2015_Calib\DScomputed';
MyCPFName = 'VNREDSAT_1_20141226_093000_20141227_000001.cpf';
MyCPF = fullfile(MyCPFPath,MyCPFName);
MyCPFTree = xml_read(MyCPF);
gen_object_display(MyCPFTree);
DS_PAN_CPF=str2num(MyCPFTree.RadiometricParameters.PAN.G1.DarkCurrents);
DS_B1_CPF=str2num(MyCPFTree.RadiometricParameters.B1.G1.DarkCurrents);
DS_B2_CPF=str2num(MyCPFTree.RadiometricParameters.B2.G1.DarkCurrents);
DS_B3_CPF=str2num(MyCPFTree.RadiometricParameters.B3.G1.DarkCurrents);
DS_B4_CPF=str2num(MyCPFTree.RadiometricParameters.B4.G1.DarkCurrents);

%Read PRNU from old CPF files (only used as reference PRNU to update the
%CPF files if needed-no change to the PRNU data)
PRNU_PAN_CPF=str2num(MyCPFTree.RadiometricParameters.PAN.G1.DetectorRelativeGains);
PRNU_B1_CPF=str2num(MyCPFTree.RadiometricParameters.B1.G1.DetectorRelativeGains);
PRNU_B2_CPF=str2num(MyCPFTree.RadiometricParameters.B2.G1.DetectorRelativeGains);
PRNU_B3_CPF=str2num(MyCPFTree.RadiometricParameters.B3.G1.DetectorRelativeGains);
PRNU_B4_CPF=str2num(MyCPFTree.RadiometricParameters.B4.G1.DetectorRelativeGains);


%3. Extract the low frequency part from the ref CPF
% sigma = 5;
% DS_PAN_LF = mean(DS_PAN_CPF_ref)*DS_PAN_CPF_ref./lowPassFilter(DS_PAN_CPF,sigma);
% DS_B1_LF = mean(DS_B1_CPF_ref)*DS_B1_CPF_ref./lowPassFilter(DS_B1_CPF,sigma);
% DS_B2_LF = mean(DS_B2_CPF_ref)*DS_B2_CPF_ref./lowPassFilter(DS_B2_CPF,sigma);
% DS_B3_LF = mean(DS_B3_CPF_ref)*DS_B3_CPF_ref./lowPassFilter(DS_B3_CPF,sigma);
% DS_B4_LF = mean(DS_B4_CPF_ref)*DS_B4_CPF_ref./lowPassFilter(DS_B4_CPF,sigma);

sigmaPAN =40;
sigmaMS = 10;
DS_PAN_ref_LF = lowPassFilter(DS_PAN_CPF_ref,sigmaPAN);
DS_B1_ref_LF = lowPassFilter(DS_B1_CPF_ref,sigmaMS);
DS_B2_ref_LF = lowPassFilter(DS_B2_CPF_ref,sigmaMS);
DS_B3_ref_LF = lowPassFilter(DS_B3_CPF_ref,sigmaMS);
DS_B4_ref_LF = lowPassFilter(DS_B4_CPF_ref,sigmaMS);


%4. Add the Low frequency part from the ref CPF to the newly computed CPF
DS_PAN_CPF_final = (DS_PAN_CPF./lowPassFilter(DS_PAN_CPF,sigmaPAN)).*DS_PAN_ref_LF;
DS_B1_CPF_final = (DS_B1_CPF./lowPassFilter(DS_B1_CPF,sigmaMS)).*DS_B1_ref_LF;
DS_B2_CPF_final = (DS_B2_CPF./lowPassFilter(DS_B2_CPF,sigmaMS)).*DS_B2_ref_LF;
DS_B3_CPF_final = (DS_B3_CPF./lowPassFilter(DS_B3_CPF,sigmaMS)).*DS_B3_ref_LF;
DS_B4_CPF_final = (DS_B4_CPF./lowPassFilter(DS_B4_CPF,sigmaMS)).*DS_B4_ref_LF;
% DS_PAN_CPF = DS_PAN_CPF + DS_PAN_ref_LF;
% DS_B1_CPF = DS_B1_CPF + DS_B1_ref_LF;
% DS_B2_CPF = DS_B2_CPF + DS_B2_ref_LF;
% DS_B3_CPF = DS_B3_CPF + DS_B3_ref_LF;
% DS_B4_CPF = DS_B4_CPF + DS_B4_ref_LF;

%5. Calculate the difference after straylight process
DS_PAN_Diff = DS_PAN_CPF_final - DS_PAN_CPF_ref;
DS_B1_Diff = DS_B1_CPF_final - DS_B1_CPF_ref;
DS_B2_Diff = DS_B2_CPF_final - DS_B2_CPF_ref;
DS_B3_Diff = DS_B3_CPF_final - DS_B3_CPF_ref;
DS_B4_Diff = DS_B4_CPF_final - DS_B4_CPF_ref;

%6. plot all the DS for comparison
figure;
subplot(5,2,1); plot(DS_PAN_CPF); hold on;plot(DS_PAN_CPF_final,'y'); hold on; plot(DS_PAN_CPF_ref,'g'); hold on; plot(DS_PAN_ref_LF,'r'); title('Compare DS PAN: Blue:original, Yellow:new computed, Green: reference, Red: LF part of reference');
subplot(5,2,2); plot(DS_PAN_Diff);title('DS PAN different in LSB');
subplot(5,2,3); plot(DS_B1_CPF); hold on; plot(DS_B1_CPF_final,'y'); hold on; plot(DS_B1_CPF_ref,'g');hold on; plot(DS_B1_ref_LF,'r'); title('Compare DS B1: Blue:original, Yellow:new computed, Green: reference, Red: LF part of reference');
subplot(5,2,4); plot(DS_B1_Diff);title('DS B1 different in LSB');
subplot(5,2,5); plot(DS_B2_CPF); hold on;plot(DS_B2_CPF_final,'y'); hold on; plot(DS_B2_CPF_ref,'g');hold on; plot(DS_B2_ref_LF,'r'); title('Compare DS B2: Blue:original, Yellow:new computed, Green: reference, Red: LF part of reference');
subplot(5,2,6); plot(DS_B2_Diff);title('DS B2 different in LSB');
subplot(5,2,7); plot(DS_B3_CPF); hold on;plot(DS_B3_CPF_final,'y'); hold on; plot(DS_B3_CPF_ref,'g');hold on; plot(DS_B3_ref_LF,'r'); title('Compare DS B3: Blue:original, Yellow:new computed, Green: reference, Red: LF part of reference');
subplot(5,2,8); plot(DS_B3_Diff);title('DS B3 different in LSB');
subplot(5,2,9); plot(DS_B4_CPF); hold on;plot(DS_B4_CPF_final,'y'); hold on; plot(DS_B4_CPF_ref,'g');hold on; plot(DS_B4_ref_LF,'r'); title('Compare DS B4: Blue:original, Yellow:new computed, Green: reference, Red: LF part of reference');
subplot(5,2,10); plot(DS_B4_Diff);title('DS B4 different in LSB');

figure;
subplot(5,5,1); plot(DS_PAN_CPF_ref); title('DS_PAN reference');
subplot(5,5,2); plot(DS_PAN_ref_LF); title('DS_PAN reference, LF part');
subplot(5,5,3); plot(DS_PAN_CPF,'g'); title('DS_PAN new computed');
subplot(5,5,4); plot(DS_PAN_CPF_final,'g'); title('DS_PAN new computed add LF part');
subplot(5,5,5); plot(DS_PAN_Diff,'r'); title('DS_PAN diff');

subplot(5,5,6); plot(DS_B1_CPF_ref); title('DS_B1 reference');
subplot(5,5,7); plot(DS_B1_ref_LF); title('DS_B1 reference, LF part');
subplot(5,5,8); plot(DS_B1_CPF,'g'); title('DS_B1 new computed');
subplot(5,5,9); plot(DS_B1_CPF_final,'g'); title('DS_B1 new computed add LF part');
subplot(5,5,10); plot(DS_B1_Diff,'r'); title('DS_B1 diff');

subplot(5,5,11); plot(DS_B2_CPF_ref); title('DS_B2 reference');
subplot(5,5,12); plot(DS_B2_ref_LF); title('DS_B2 reference, LF part');
subplot(5,5,13); plot(DS_B2_CPF,'g'); title('DS_B2 new computed');
subplot(5,5,14); plot(DS_B2_CPF_final,'g'); title('DS_B2 new computed add LF part');
subplot(5,5,15); plot(DS_B2_Diff,'r'); title('DS_B2 diff');

subplot(5,5,16); plot(DS_B3_CPF_ref); title('DS_B3 reference');
subplot(5,5,17); plot(DS_B3_ref_LF); title('DS_B3 reference, LF part');
subplot(5,5,18); plot(DS_B3_CPF,'g'); title('DS_B3 new computed');
subplot(5,5,19); plot(DS_B3_CPF_final,'g'); title('DS_B3 new computed add LF part');
subplot(5,5,20); plot(DS_B3_Diff,'r'); title('DS_B3 diff');

subplot(5,5,21); plot(DS_B4_CPF_ref); title('DS_B4 reference');
subplot(5,5,22); plot(DS_B4_ref_LF); title('DS_B4 reference, LF part');
subplot(5,5,23); plot(DS_B4_CPF,'g'); title('DS_B4 new computed');
subplot(5,5,24); plot(DS_B4_CPF_final,'g'); title('DS_B4 new computed add LF part');
subplot(5,5,25); plot(DS_B4_Diff,'r'); title('DS_B4 diff');

%7. Write the DS after straylight process to CPF files
% if max(abs(DS_PAN_Diff)) > 0
%     bandName='PAN';
%     DSnew_PAN=DS_PAN_CPF; 
%     MyCPFnew = UpdateRadiometricParametersInCPF(MyCPF, bandName, PRNU_PAN_CPF, DSnew_PAN);
% end
% 
% if max(abs(DS_B1_Diff)) > 0
%     bandName='B1';
%     DSnew_B1=DS_B1_CPF; 
%     MyCPFnew = UpdateRadiometricParametersInCPF(MyCPFnew, bandName, PRNU_B1_CPF, DSnew_B1);
% end
% 
% if max(abs(DS_B2_Diff)) > 0
%     bandName='B2';
%     DSnew_B2=DS_B2_CPF; 
%     MyCPFnew = UpdateRadiometricParametersInCPF(MyCPFnew, bandName, PRNU_B2_CPF, DSnew_B2);
% end
% 
% if max(abs(DS_B3_Diff)) > 0
%     bandName='B3';
%     DSnew_B3=DS_B3_CPF; 
%     MyCPFnew = UpdateRadiometricParametersInCPF(MyCPFnew, bandName, PRNU_B3_CPF, DSnew_B3);
% end
% 
% if max(abs(DS_B4_Diff)) > 0
%     bandName='B4';
%     DSnew_B4=DS_B4_CPF; 
%     MyCPFnew = UpdateRadiometricParametersInCPF(MyCPFnew, bandName, PRNU_B4_CPF, DSnew_B4);
% end

