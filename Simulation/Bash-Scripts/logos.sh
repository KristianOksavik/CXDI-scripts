beamtime=ma5981
sample=logo_7um

matrix_size=1024
num=0001
scan=09
fname=${sample}_${num}

#Center
cnt0=924  
cnt1=1288  

#Pixel size
pix=75e-6

#Sample-detector distance
zsd=7.05

#Add python paths (or add python to run .py scripts)
# export PYTHONPATH=$PYTHONPATH:/data/id10/inhouse/Programs/cdi
# export PATH=$PATH:/home/daniyaly/cdiscripts/
export PATH=$PATH:/cluster/home/kristaok/reconstruction/


#Data file path
#datapath="/cluster/home/kristaok/DATA/ma5981/apertures/apertures_0001/scan0016/eiger4m_0000.h5"
#datapath=/cluster/home/kristaok/DATA/ma5981/logo_7um/logo_7um_0001/scan00${scan}/eiger4m_0000.h5
datapath=/cluster/home/kristaok/DATA/ma5981/${sample}/${fname}/scan00${scan}/eiger4m_0000.h5
#datapath="/cluster/home/kristaok/reconstruction/data_logo_filtered.npy"

##tmp analysis
#PYOPENCL_COMPILER_OUTPUT=1 HDF5_USE_FILE_LOCKING=FALSE python3 cdi-regrid_yuriy.py ${datapath} ${datapath2} ${datapath3} ${datapath4} -o ${fname}_1024.npy -m eiger4m_mask_full.npy  -s 1024 -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix}
#PYOPENCL_COMPILER_OUTPUT=1 HDF5_USE_FILE_LOCKING=FALSE python3 cdi-regrid_yuriy.py ${datapath} -o ${fname}_1024.npy -m bmask_${fname}.npy  -s 1024 -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix}

#chmod +x cdi-regrid_yuriy.py

#1. Sum data
#echo "Summing up the data"
#python3 cdi-regrid_yuriy.py ${datapath} -o ${fname}_${matrix_size}.npy -s ${matrix_size} -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix}

#2. Make mask
#echo "Make mask & save mask to: /home/reconstruction/bmask.npy"
#silx view data_to_make_bmask.npy
#save mask to: /data/id10/inhouse/Programs/cdi/bmask.npy @ESRF

#3. Add Eiger mask to: bmask
#echo "Updating mask"
#cp bmask.npy original_bmask_${fname}.npy
#mv bmask.npy bmask_${fname}.npy
#python3 final_mask.py bmask_${fname}_scan${scan}.npy eiger4m_mask_full.npy

#0 Check correlation
#echo "Checking zero"
#python3 check_zero_h5.py ${datapath} bmask_${fname}.npy ${cnt0} ${cnt1}


#4. Assemble 3D data
#echo "Gridding the data"
#python3 cdi-regrid_yuriy.py ${datapath} -o ${fname}_${matrix_size}.npy -s ${matrix_size} -m bmask_${fname}.npy -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix} --oversampling-rot 1

#PYOPENCL_COMPILER_OUTPUT=1 HDF5_USE_FILE_LOCKING=FALSE python3 cdi-regrid_yuriy.py ${datapath} ${datapath2} -o ${fname}_new_1024.npy -s 1024 -m bmask_${fname}.npy -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix} --oversampling-rot 1

#4b Diffraction data
#echo "Extract diffraction data"
#PYOPENCL_COMPILER_OUTPUT=1 HDF5_USE_FILE_LOCKING=FALSE dif_yuriy.py ${datapath} -o ${fname}_1024.npy -s 1024 -m bmask_${fname}.npy -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix}
#
#4c Structure 2D data
#echo "Extracting 2D data"
python3 Analyze2D.py ${datapath} -o ${fname}_${matrix_size}_scan${scan}.npy -s ${matrix_size} -m ESRF/logo_7um_0001/bmask_logo_7um_0001.npy -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix}

##5. Inspect data
#echo "Regridded Data"
#silx view ${fname}_${matrix_size}*.npy

##5b. Cleaning of Data
echo "Auto-removing hot pixels"
python3 auto_remove_hot_3D.py ${fname}_${matrix_size}_scan${scan}.npy
#echo "Remove hot pixels by thresholding each frame and save"
#python3 remove_hot_3D.py ${fname}_${matrix_size}.npy

##6. Make analysis directory
#echo "Making directory for data"
#mkdir /cluster/home/kristaok/reconstruction/ESRF/${fname}

##7. Copy data (numpy files) into analysis directory
#echo "Moving data."
#mv *${fname}*.npy /cluster/home/kristaok/reconstruction/ESRF/${fname}/.
#mv *${fname}*.npy "/cluster/work/kristaok/logo_7um_0001/filtered/"
#mkdir ${fname}/recon_scan16
#rm data_to_make_bmask.npy

#8. Path for reconstructed files
#echo "New subfolder for reconstruction .cxi files"
#mkdir /cluster/home/kristaok/reconstruction/ESRF/${fname}/${fname}
#mkdir /cluster/home/kristaok/reconstruction/ESRF/${fname}/${fname}_cxifiles95

#loop
#for value in {140..200}
#do
#    printf -v i "%03d" ${value}
#    echo "${i} of 200"
#    pynx-id10cdi.py data=${fname}/${fname}_frames/${fname}_${matrix_size}_split_010.npy support_type=circle support_size=${value} support_threshold=0.06,0.2 support_threshold_method=max max_size=1024 live_plot=False nb_run=200 nb_run_keep=20 algorithm="ML**100,ER**300,(HIO**4000)"

#    pynx-cdi-analysis.py *LLKf* modes=2 > ${fname}/${fname}_cxifiles95.out
    
#    rm *.cxi
#    mv *modes* ${fname}_averaged_modes_support_${i}.h5
#    mv *modes* /home/kristianos/ESRF/${fname}/.
#done

# DO A SPLIT HERE WITH "split2D.py"

#9. Reconstruction of data (_averaged or _split_xxx) [scan 12 is without logo, _frames is with logo]
#pynx-cdi-id10 --data=/cluster/home/kristaok/reconstruction/ESRF/${fname}/${fname}_${matrix_size}_scan${scan}_auto_clean.npy --support /cluster/home/kristaok/reconstruction/ESRF/${fname}/support_${fname}_${matrix_size}_scan${scan}.npy --support_threshold 0.09 0.1 --support_threshold_method rms --max_size ${matrix_size} --verbose 0 --nb_run 50 --nb_run_keep 20 --algorithm "ER**300,(sup*ER**100*HIO**1000)**20" 


#9a. Reconstruction of every split frame
#loop

#mkdir ${fname}/recon_scan13
#for value in {1..51}
#do
#    printf -v i "%03d" ${value}
#    echo "${i} of 51"
#pynx-cdi-id10 --data /cluster/home/kristaok/reconstruction/ESRF/${fname}/${fname}_scan16/${fname}_split_026.npy --support="/cluster/home/kristaok/reconstruction/support_8umaperture.npy" --support_threshold 0.08 0.1 --support_threshold_method max --max_size 1024 --verbose 0 --nb_run 300 --nb_run_keep 30 --algorithm "ML**100,ER**300,(sup*HIO**300)**20"
#    pynx-id10cdi.py --data ${fname}/${fname}_frames/${fname}_split_${i}.npy --support=circle --support_size 115 --support_threshold=0.06 --support_threshold_method max --max_size 512 --verbose 300 --nb_run 300 --nb_run_keep 30 --algorithm "ML**100,ER**300,(sup*HIO**30)**100"
#    pynx-cdi-analysis.py *LLKf* modes=2 > ${fname}/${fname}_cxifiles95.out
#    rm *.cxi
#    mv *modes* /home/kristianos/ESRF/${fname}/recon_scan13/.
#done
    
#10. Averaging of cxi
#echo "Averaging .cxi files"
#pynx-cdi-analysis.py *LLKf* modes=2 > /cluster/home/kristaok/reconstruction/ESRF/${fname}/${fname}_cxifiles95.out

#11. Moving into corresponding folder
#mv *.cxi /cluster/work/kristaok/${fname}/cxibackup/
#mv *modes* /cluster/work/kristaok/${fname}/processed/
#echo "Moving all files to folders" 

rm *.cxi

echo "-----------------------------PROCESS FINISHED-----------------------------"
