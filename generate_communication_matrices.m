
function communication_waytotal(walker)

addpath('C:\Users\umrzahnef\Documents\BCT\2019_03_03_BCT');

subdir  = 'F:\all_mats';
matsdir = fullfile(subdir, 'raw_mats');
distance_dir = 'F:\D_mats';

switch walker
    case 'navigation'
        source_dir = 'F:\Communication\waytotal\';
        output_dir = 'F:\Communication\waytotal\Navigation';
    case 'diffusion'
        source_dir = 'F:\Communication\waytotal\';        
        output_dir = 'F:\Communication\waytotal\Diffusion_Efficiency';
    case 'searchinformation'
        source_dir = 'F:\Communication\waytotal\';        
        output_dir = 'F:\Communication\waytotal\Search_Information';
end

dryrun = false;

threshlist = {'0','80'};
verbose = true; % use this to display some info

%eval(['!dir ',matsdir]);
folders = dir(matsdir); 

sublist = {};
% loop over dirs, start at 3 bc 1 and 2 are "." and ".." 
for dirnum = 3:size(folders,1)
    foldername = folders(dirnum).name;
    sub = foldername(1:6);
    if ~any(ismember(sublist, sub)) & ~strcmp(sub, 'Columns')
        sublist{end+1} = sub;
    end
end


for threshnum = 1:length(threshlist)  
    thresh = threshlist{threshnum};
    for subnum = 1:length(sublist)
        sub = sublist{subnum};
        
        if verbose; 
            disp(['Now processing subject ' ...
                 , num2str(subnum),' of ', num2str(length(sublist))]); 
        end        
        
        switch walker
        case 'navigation'
            m_file  = fullfile(source_dir, [sub, '_invmat_allroi_'  , thresh, '.txt']);
            n_file  = fullfile(distance_dir, [sub, '_lengths']);
            m = dlmread(m_file);
            n = dlmread(m_file);
            outfile   = fullfile(output_dir, [sub, '_navigation_wei_', thresh, '.txt']);
            outfile_1 = fullfile(output_dir, [sub, '_navigation_bin_', thresh, '.txt']);
            outfile_2 = fullfile(output_dir, [sub, '_sr_nav_'        , thresh, '.txt']);            
            [sr, PL_bin, PL_wei, PL_dis, paths] = navigation_wu(m, n);
            dlmwrite(outfile, PL_wei);       
            dlmwrite(outfile_1, PL_bin); 
            dlmwrite(outfile_2, sr);                     
        case 'diffusion'
            m_file  = fullfile(source_dir, [sub, '_m_normforinvert'  , thresh, '.txt']);
            m = dlmread(m_file);
            outfile   = fullfile(output_dir, [sub, '_GEdiff_', thresh, '.txt']);
            outfile_1 = fullfile(output_dir, [sub, '_Ediff_' , thresh, '.txt']);
            [GEdiff,Ediff] = diffusion_efficiency(m);
            dlmwrite(outfile  , GEdiff);
            dlmwrite(outfile_1, Ediff );
        case 'searchinformation'
            m_file  = fullfile(source_dir, [sub, '_invmat_allroi_'  , thresh, '.txt']);            
            n_file  = fullfile(source_dir, [sub, '_m_normforinvert'  , thresh, '.txt']);
            m = dlmread(m_file);
            n = dlmread(n_file);
            SI = search_information(n, m, true);
            outfile = fullfile(output_dir, [sub, '_SI_', thresh, '.txt']);
            dlmwrite(outfile, SI);
        end
                
    end
end
