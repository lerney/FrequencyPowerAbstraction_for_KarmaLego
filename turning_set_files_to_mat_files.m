
% Specify the input and output folder paths
inputFolder = "C:\\final_project\\1024hz sampled H1H7";
outputFolder = "C:\\final_project\\1024hz sampled H1H7 mat files";

% Get a list of all .set files in the input folder and its subfolders
fileList = getAllFiles(inputFolder, '.set');

% Loop through each .set file
for i = 1:length(fileList)
    % Load the EEG dataset using EEGlab
    EEG = pop_loadset('filename', fileList{i}, 'filepath', '');

    % Save the EEG data as a MATLAB .mat file in the output folder
    [~, fileName, ~] = fileparts(fileList{i});
    outputFilePath = fullfile(outputFolder, [fileName '.mat']);
    save(outputFilePath, 'EEG');
end

% Function to recursively get all files with a given extension
function fileList = getAllFiles(dirName, extension)
    fileList = {};
    files = dir(dirName);
    
    for i = 1:length(files)
        if files(i).isdir
            if ~strcmp(files(i).name, '.') && ~strcmp(files(i).name, '..')
                subdir = fullfile(dirName, files(i).name);
                fileList = [fileList; getAllFiles(subdir, extension)];
            end
        else
            [~, ~, fileExt] = fileparts(files(i).name);
            if strcmp(fileExt, extension)
                fileList = [fileList; fullfile(dirName, files(i).name)];
            end
        end
    end
end