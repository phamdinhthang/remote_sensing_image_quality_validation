DS_folder = 'C:\Users\admin\Desktop\DS_PRNU\ASCENDING';
PRNU_folder = 'C:\Users\admin\Desktop\DS_PRNU\DESCENDING';


folder_array = {DS_folder,PRNU_folder};
for index=1:length(folder_array)
    folder = folder_array{index};
    subfolders = dir(folder);
    for k = 3:length(subfolders) 
        currDir = subfolders(k).name;
        full_path = fullfile(folder,currDir);
        convert_raw_to_hdf5(full_path);
    end
end

