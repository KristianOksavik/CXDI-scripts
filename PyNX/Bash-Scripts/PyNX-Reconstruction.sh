beamtime=ma5103
sample=13Da_cand1
matrix_size=768
num=0014
fname=${sample}_${num}
#Center
cnt0=1090  
cnt1=1288  

#Pixel size
pix=75e-6

#Sample-detector distance
zsd=5.45

#Add python paths (or add python to run .py scripts)
# export PYTHONPATH=$PYTHONPATH:/data/id10/inhouse/Programs/cdi
# export PATH=$PATH:/home/daniyaly/cdiscripts/

#Data file path
# datapath=/media/nfs/qnap/home/daniyaly/esrfdata/${beamtime}/id10/${sample}/${fname}/${fname}.h5
datapath=/cluster/home/kristaok/DATA/${fname}/${fname}.h5
#datapath=/media/nfs/qnap/home/kristianos/cdi-data-id10/ma5103/K1_cand4/K1_cand4_0001/K1_cand4_0001.h5


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
#python3 final_mask.py bmask_${fname}.npy eiger4m_mask_full.npy

#0 Check correlation
#echo "Checking zero"
#python3 check_zero_h5.py ${datapath} bmask_${fname}.npy ${cnt0} ${cnt1}


#4. Assemble 3D data
#echo "Gridding the data"
#python3 cdi-regrid_yuriy.py ${datapath} -o ${fname}_${matrix_size}.npy -s ${matrix_size} -m "/cluster/home/kristaok/reconstruction/ESRF/K1_final_0001/bmask_K1_final_0001.npy" -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix} --oversampling-rot 1


#4b Diffraction data
#echo "Extract diffraction data"
#python3 dif_yuriy.py ${datapath} -o ${fname}_1024.npy -s 1024 -m "/cluster/home/kristaok/reconstruction/ESRF/K1_final_0001/bmask_K1_final_0001.npy" -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix}
#PYOPENCL_COMPILER_OUTPUT=1 HDF5_USE_FILE_LOCKING=FALSE dif_yuriy.py ${datapath} -o ${fname}_1024.npy -s 1024 -m bmask_${fname}.npy -d ${zsd} -b ${cnt0} ${cnt1} -p ${pix}
#
#
##5. Inspect data
#echo "Regridded Data"
#silx view ${fname}_${matrix_size}.npy

##5b. Cleaning of Data
#echo "Remove hot pixels by thresholding each frame and save"
#python3 remove_hot_3D.py ${fname}_${matrix_size}.npy
#echo "Auto-removing hot pixels"
#python3 auto_remove_hot_3D.py ${fname}_${matrix_size}.npy

##6. Make analysis directory
#echo "Making directory for data"
#mkdir /cluster/home/kristaok/reconstruction/ESRF/${fname}
#mkdir /cluster/home/kristaok/reconstruction/ESRF/${fname}/${fname}_cxifiles95
#mkdir /cluster/work/kristaok/${fname}
#mkdir /cluster/work/kristaok/${fname}/cxibackup
#mkdir /cluster/work/kristaok/${fname}/processed

##7. Copy data (numpy files) into analysis directory
#echo "Moving data."
#mv *${fname}*.npy /cluster/home/kristaok/reconstruction/ESRF/${fname}/.
 
#rm data_to_make_bmask.npy

#9. Reconstruction of data
#pynx-id10cdi.py data=${fname}/half_${fname}_${matrix_size}_auto_clean.npy support_type=circle support_size=150 support_threshold=0.05,0.08 support_threshold_method=max max_size=512 live_plot=False nb_run=300 nb_run_keep=30 algorithm="ML**100,ER**300,(sup*HIO**200)**20"
#mpiexec -n 1 pynx-cdi-id10 --data=/cluster/home/kristaok/reconstruction/ESRF/${fname}/${fname}_${matrix_size}_auto_clean.npy --support "/cluster/home/kristaok/reconstruction/ESRF/13Da_cand1_0014/support_13Da_cand1_768.npy" --support_threshold 0.033 0.034 --support_threshold_method max --verbose 1000 --max_size=768 --mpi scan --nb_run=100 --nb_run_keep=50 --tv_er 0.01 --tv_hio 0.2 --algorithm "ER**300,(sup*(ER**100*HIO**500)**10"


#mpiexec -n 1 pynx-cdi-id10 --data=/cluster/home/kristaok/reconstruction/ESRF/${fname}/${fname}_${matrix_size}_auto_clean.npy --support "/cluster/home/kristaok/reconstruction/ESRF/K1_cand4_0001/support_K1_cand4_768.npy" --support_threshold 0.01 0.02 --support_threshold_method max --verbose 0 --max_size=768 --mpi scan --nb_run=100 --nb_run_keep=50 --tv_er 0.005 --tv_hio 0.2 --algorithm "ER**500,(sup*ER**50*HIO**500)**10"

#10. Averaging of cxi

#echo "Averaging cxi files"
#pynx-cdi-analysis *LLKf* modes=2 > ESRF/${fname}/${fname}_cxifiles95.out

#11. Moving into corresponding folder
#mv *.cxi /cluster/work/kristaok/${fname}/cxibackup/.
#mv *modes* /cluster/work/kristaok/${fname}/processed/.
#echo "Moving all files to folders"



echo "-----------------------------PROCESS FINISHED-----------------------------"