#! /bin/tcsh -f

if($#argv == 0) then
  echo "usage: desikan_gen_seeds <diffusion-dir>"
  echo "expects aseg2diff in diffusion-dir"
  exit
endif

set work_dir = $1
set seeds_file = "/home/felix/neuro-bin/language/seed_list.txt"

#Checking...
if (! -e $work_dir/aseg2diff.nii.gz) then 
  echo "aseg2diff missing"
  exit
endif


# setenv SUBJECTS_DIR $work_dir

#Generating Targets...
cat $seeds_file | awk -v aparc=$work_dir/aseg2diff.nii.gz -v output=$work_dir/seeds '{print "mri_binarize --i "aparc" --match "$1" --o "output"/"$2".nii.gz"}' | tcsh
cat $seeds_file | awk -v output=$work_dir/seeds '{print output"/"$2".nii.gz"}' > $work_dir/seeds/seeds.txt
