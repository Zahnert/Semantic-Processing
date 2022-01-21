#! /bin/tcsh -f

if($#argv == 0) then
  echo "usage: create_exclusion_mask <diffusion_dir>"
  exit
endif

set diffusion_dir = $1

mkdir /tmp/brain_wo_ventricles
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 4 -uthr 4 /tmp/brain_wo_ventricles/aseg-seg4.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 5 -uthr 5 /tmp/brain_wo_ventricles/aseg-seg5.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 43 -uthr 43 /tmp/brain_wo_ventricles/aseg-seg43.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 44 -uthr 44 /tmp/brain_wo_ventricles/aseg-seg44.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 14 -uthr 14 /tmp/brain_wo_ventricles/aseg-seg14.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 15 -uthr 15 /tmp/brain_wo_ventricles/aseg-seg15.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 24 -uthr 24 /tmp/brain_wo_ventricles/aseg-seg24.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 30 -uthr 30 /tmp/brain_wo_ventricles/aseg-seg30.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 62 -uthr 62 /tmp/brain_wo_ventricles/aseg-seg62.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 31 -uthr 31 /tmp/brain_wo_ventricles/aseg-seg31.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 63 -uthr 63 /tmp/brain_wo_ventricles/aseg-seg63.nii.gz
fslmaths $diffusion_dir/aseg2diff.nii.gz -thr 72 -uthr 72 /tmp/brain_wo_ventricles/aseg-seg72.nii.gz

fslmaths /tmp/brain_wo_ventricles/aseg-seg4.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg5.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg43.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg44.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg14.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg15.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg24.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg30.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg62.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg31.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg63.nii.gz -add /tmp/brain_wo_ventricles/aseg-seg72.nii.gz /tmp/brain_wo_ventricles/exclusion.nii.gz
fslmaths /tmp/brain_wo_ventricles/exclusion.nii.gz -bin $diffusion_dir/exclusion_mask.nii.gz
rm -rf /tmp/brain_wo_ventricles
