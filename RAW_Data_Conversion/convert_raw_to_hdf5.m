function convert_raw_to_hdf5(full_folder_path)
disp(full_folder_path)
if isempty(strfind(full_folder_path, 'MS'))
    metadataFile='METADATA.L0';
    L0info=ReadL0Metadata(fullfile(full_folder_path,metadataFile));

    bandName ='PAN';
    l_end = L0info.nRows;
    disp(['nrows=',num2str(l_end)])
    PAN_L0_Image = ReadL0Image(bandName, full_folder_path, 1, l_end);
%     figure; 
%     imagesc(PAN_L0_Image); colormap gray; colorbar; title(strcat(bandName,' raw data'));
    hdf5write(fullfile(full_folder_path,strcat(bandName,'.h5')), bandName, PAN_L0_Image);
else
    metadataFile='METADATA.L0';
    L0info=ReadL0Metadata(fullfile(full_folder_path,metadataFile));
    l_end = L0info.nRows;
    disp(['nrows=',num2str(l_end)])
    MS_bands = {'B1','B2','B3','B4'};
    for index=1:length(MS_bands)
        bandName = MS_bands{index};
        MS_L0_Image = ReadL0Image(bandName, full_folder_path, 1, l_end);
%         figure; 
%         imagesc(MS_L0_Image); colormap gray; colorbar; title(strcat(bandName,' raw data'));
        hdf5write(fullfile(full_folder_path,strcat(bandName,'.h5')), bandName, MS_L0_Image);
    end
end
end