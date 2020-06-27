import pylidc
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours

import pylidc as pl
from pylidc.utils import consensus

import nibabel as nib

DATA_PATH = "/mnt/ed1b201c-adcd-4fbb-a01a-5f3d84b9a1ed/LIDC-IDRI/LIDC_IDRI_AS_NIFTI/"

def save_to_nifti(matrix, info, folder_name, filename):
   
    folder_path = os.path.join(DATA_PATH,folder_name)

    if not(os.path.exists(folder_path)):
        os.mkdir(folder_path)

    img_nifti = nib.Nifti1Image(np.rot90(matrix), None)#np.swapaxes(np.fliplr(matrix), 0, 2), None)
    img_nifti.header.set_zooms(info['spacing'])
    img_nifti.header.set_dim_info(slice=0)
    nib.save(img_nifti, os.path.join(folder_path,filename))


def save_txt_info(filename, data):
    f = open(filename, "w")
    f.write(data)
    f.close()


def plot_concesus():
    # Query for a scan, and convert it to an array volume.
    
    scans = pl.query(pl.Scan)
    data = ""

    #c = 0

    for scan in scans:
        q = pl.query(pl.Scan).filter(pl.Scan.patient_id == scan.patient_id).first()

        info = {}
        info['spacing'] = [q.pixel_spacing, q.pixel_spacing, q.slice_spacing]

        vol = scan.to_volume()
        vol_mask = np.zeros(vol.shape).astype('bool')

        # Cluster the annotations for the scan, and grab one.
        nods = scan.cluster_annotations()
        
        print(q.patient_id, "\n")

        data += q.patient_id
        coordinates = ""
        
        for anns in nods:
            
            #anns = nods[0]
            

            # Perform a consensus consolidation and 50% agreement level.
            # We pad the slices to add context for viewing.
            cmask,cbbox,masks = consensus(anns, clevel=0.6,
                                          pad=[(20,20), (20,20), (0,0)])
            
            #print(cbbox[0], cbbox)

            xf = cbbox[0].stop
            xi = cbbox[0].start

            yf = cbbox[1].stop
            yi = cbbox[1].start

            zf = cbbox[2].stop
            zi = cbbox[2].start
            
            coordinates += ":" + str(xi) + "," + str(xf) + "," + str(yi) + "," +\
                    str(yf) + "," + str(zi) + "," + str(zf)
            
            # Get the central slice of the computed bounding box.
            #k = int(0.5*(cbbox[2].stop - cbbox[2].start))
           
            vol_mask[xi:xf,yi:yf,zi:zf] = cmask#[:,:,:].astype(float)


            
            #plt.imshow(vol[xi:xf,yi:yf,zi], cmap=plt.cm.Greys_r)
            #plt.imshow(vol_mask[:,:,zi + 2], cmap=plt.cm.Greys_r)#.astype(float))
            
            #plt.imshow(vol[cbbox][:,:,k], cmap=plt.cm.Greys_r)
            #vol_mask[cbbox][:,:,k] = cmask[:,:,k].astype(float)
            #plt.imshow(vol_mask[:,:,k], cmap=plt.cm.Greys_r)
            #plt.show()
       
            #print('\n',q.patient_id, xi, xf, yi, yf, zi, zf)

        data += coordinates + "\n"

        #plt.imshow(vol[xi:xf,yi:yf,zi], cmap=plt.cm.Greys_r)
        #plt.show()


        save_to_nifti(vol_mask.astype(float), info, q.patient_id, "mask.nii.gz")
        save_to_nifti(vol.astype("int16"), info, q.patient_id, "ct.nii.gz")
        #break
        

        #c += 1

        #if c == 10:
        #    break

    save_txt_info("nodules_list.txt", data)

def read_directory(folder):
    pass

if __name__ == '__main__':

    #main_folder = sys.argv[1]
    #read_directory(main_folder)

    plot_concesus()
