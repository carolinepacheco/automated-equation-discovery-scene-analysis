import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def compute_LBP(imgnm, lbp_str): 
    image = plt.imread(imgnm)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgLBP = np.zeros_like(gray_image) 
    neighboor = 3 
    s = 0
    for ih in range(0,image.shape[0] - neighboor): # loop by line
        for iw in range(0,image.shape[1] - neighboor):
            ### Get natrix image 3 by 3 pixel
            a = 0.01
            Z = gray_image[ih:ih+neighboor,iw:iw+neighboor] 
            C = Z[1,1]
            
            
            Z = Z.astype(np.float32)
            C = C.astype(np.float32)
        
            s = eval(lbp_str)

            lbp = (s >= 0)*1.0
            lbp = lbp.astype(np.uint8)
            
        
            img01_vector = lbp.T.flatten()
        
            img01_vector = np.delete(img01_vector,4)
        
            ### Convert the binary operated values to a decimal (digit)
            where_lbp_vector = np.where(img01_vector)[0]

            num = np.sum(2**where_lbp_vector) if len(where_lbp_vector) >= 1 else 0
            imgLBP[ih+1,iw+1] = num
## print the LBP values when frequencies are high

            
    return image, gray_image, imgLBP


