# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:36:41 2016

@author: Lukas Gartmair
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as pl
import scipy.special as sps
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

from scipy.ndimage.filters import gaussian_filter
from skimage import measure

import time
start_time = time.time()



import matplotlib.patches as patches
################### Functions #################

def check_for_particles_in_crack(labeled_matrix, rect_xs, rect_ys):
    particle_content = []

    particles_already_counted = []    
    
    for i in rect_xs:
        for j in rect_ys:
            if labeled_matrix[i,j] != 0:
                if labeled_matrix[i,j] not in particles_already_counted:
                    particle_label = labeled_matrix[i,j]
                    particle_amount = np.where(labeled_matrix == particle_label)
                    particle_content.append(particle_amount[0].size)
                    
                    particles_already_counted.append(particle_label)
                    
    return particle_content
    

def calc_rectangle_coords(center_x, center_y, edge_a, edge_b, angle=0):
    if angle == 0:
        rect_xs = np.arange(center_x-int(edge_a/2),center_x + int(edge_a/2))
        rect_ys = np.arange(center_y-int(edge_b/2),center_y + int(edge_b/2))
    
    if angle == 90:
        rect_xs = np.arange(center_x-int(edge_b/2),center_x + int(edge_b/2))
        rect_ys = np.arange(center_y-int(edge_a/2),center_y + int(edge_a/2))
        
    
    return rect_xs, rect_ys
    
def draw_rectangle(matrix_area, rect_xs, rect_ys, crack_color):
    
    for i in rect_xs:
        for j in rect_ys:
            matrix_area[i,j] = crack_color
    return matrix_area
            

def calc_indent_coords(d1,d2,indent_shifter, center_indent_x, center_indent_y):
    vickers_xs = np.arange(center_indent_x-int(d1/2),center_indent_x + int(d1/2))
    vickers_ys = np.arange(center_indent_y-int(d2/2),center_indent_y + int(d2/2))  + indent_shifter
    
    return vickers_xs, vickers_ys
    
def vickers_indent(vickers_xs, vickers_ys, matrix_area,successfull_indent_color, indenter_color, particle_color):
    indent = False
    for i in vickers_xs:
        for j in vickers_ys:
            if matrix_area[i,j] == particle_color:
                indent  = True
                matrix_area[i,j] = successfull_indent_color
            else:
                matrix_area[i,j] = indenter_color  
    return indent

def in_sphere(center_x, center_y, center_z, radius, x, y,z):
    square_dist = ((center_x - x) ** 2 + (center_y - y) ** 2 + (center_z - z) ** 2 )
    return square_dist <= radius ** 2

def make_sphere(cx,cy,cz,radius,cxs,cys,czs):
    in_sphere_arr = np.arange(0)
    cxx, cyy, czz = np.meshgrid(cxs,cys,czs)
    in_sphere_arr = np.array(in_sphere(cx,cy,cz,radius,cxx,cyy,czz))
    return in_sphere_arr.astype(int)
    
def calc_global_overlap_coords(matrix_subset, cx, cy, cz, radius):
    xs, ys, zs = np.where(matrix_subset==1)
    xs = np.copy(xs + (cx-radius))
    ys = np.copy(ys + (cy-radius))
    zs = np.copy(zs + (cz-radius))
    return xs, ys, zs
    
def check_overlap(matrix_area_subset,matrix_color):
    overlap = True
    if np.any(matrix_area_subset!=matrix_color):
        overlap == True
    else:
        overlap = False
    return overlap

def correct_boundaries(boundary,tmp):
    # over the limit
    for c in range(tmp.size):
        if tmp[c] >= boundary:
            tmp[c] = tmp[c]-boundary
    # below the limit
        if tmp[c] < 0:
            tmp[c] = tmp[c]+boundary
    return tmp    

################### Variables #################

# edge lenghts cube
a = 400
b = 400
c = 400
volume_total = a*b*c

# volume arrays
a_vol = np.arange(a)
b_vol = np.arange(b)
c_vol = np.arange(c) 

# volume fractions
#volume_fraction_of_particles_total = 0.1
volume_fraction_of_particles_totals = [0.1]


#  color settings    
matrix_color = 0
particle_containing_color = 1
particle_color = 3
indenter_color = 4
crack_color = 6

################### DATA #####################
diameters = np.array([  1.00000000e-02,   1.10000000e-02,   1.30000000e-02,
         1.40000000e-02,   1.60000000e-02,   1.80000000e-02,
         2.10000000e-02,   2.40000000e-02,   2.70000000e-02,
         3.00000000e-02,   3.40000000e-02,   3.80000000e-02,
         4.30000000e-02,   4.90000000e-02,   5.50000000e-02,
         6.20000000e-02,   7.00000000e-02,   8.00000000e-02,
         9.00000000e-02,   1.02000000e-01,   1.15000000e-01,
         1.30000000e-01,   1.47000000e-01,   1.66000000e-01,
         1.87000000e-01,   2.11000000e-01,   2.39000000e-01,
         2.70000000e-01,   3.05000000e-01,   3.45000000e-01,
         3.89000000e-01,   4.40000000e-01,   4.97000000e-01,
         5.61000000e-01,   6.34000000e-01,   7.17000000e-01,
         8.10000000e-01,   9.15000000e-01,   1.03400000e+00,
         1.16800000e+00,   1.32000000e+00,   1.49100000e+00,
         1.68400000e+00,   1.90300000e+00,   2.15000000e+00,
         2.42900000e+00,   2.74500000e+00,   3.10100000e+00,
         3.50300000e+00,   3.95800000e+00,   4.47200000e+00,
         5.05300000e+00,   5.70900000e+00,   6.45000000e+00,
         7.28700000e+00,   8.23300000e+00,   9.30200000e+00,
         1.05100000e+01,   1.18740000e+01,   1.34160000e+01,
         1.51570000e+01,   1.71250000e+01,   1.93480000e+01,
         2.18600000e+01,   2.46980000e+01,   2.79040000e+01,
         3.15270000e+01,   3.56200000e+01,   4.02440000e+01,
         4.54690000e+01,   5.13710000e+01,   5.80410000e+01,
         6.55750000e+01,   7.40890000e+01,   8.37070000e+01,
         9.45740000e+01,   1.06852000e+02,   1.20724000e+02,
         1.36397000e+02,   1.54104000e+02,   1.74110000e+02,
         1.96714000e+02])
 
volume_percents_particles = np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.01,  0.06,  0.1 ,  0.15,  0.21,
        0.29,  0.38,  0.49,  0.62,  0.77,  0.96,  1.19,  1.46,  1.78,
        2.16,  2.6 ,  3.08,  3.61,  4.17,  4.72,  5.24,  5.68,  6.  ,
        6.17,  6.16,  5.97,  5.62,  5.13,  4.54,  3.91,  3.28,  2.7 ,
        2.2 ,  1.78,  1.45,  1.19,  1.  ,  0.84,  0.71,  0.59,  0.45,
        0.33,  0.19,  0.06,  0.01,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ])
        

##################  Calculations  ###############

radii = []

summary_total = []

for volume_fraction_of_particles_total in volume_fraction_of_particles_totals:
    
    radii = []
    radii = diameters / 2

    volume_of_particles_total = volume_total*volume_fraction_of_particles_total
    
    # calc how much volume belongs to each particle radius from the distribution
    volume_of_each_particle_radius = volume_of_particles_total * (volume_percents_particles/100)
    
    # calc how many spheres of each radius are needed to fill the volume reserved for each particle radius
    number_of_particles = volume_of_each_particle_radius / ((4/3)*(np.pi * (radii**3)))
    
    # round the particle radii for further processing
    
    radii = np.round(radii,decimals=0)
    
    # fill an array with each radius times its frequency
    
    radii_list = []
    for i,r in enumerate(radii):
        for j in range(int(np.round(number_of_particles[i]))):
            radii_list.append(r)
            
    radii_final = np.array(radii_list)
    
    ##################  Volume #######################
    
    matrix_vol = np.zeros((a,b,c))    
    
    ## place the big ones first, then the small ones this will save a lot of time and money
     
    radii_sorted = np.sort(radii_final)[::-1]
     
    for i,radius in enumerate(radii_sorted):  
    
            flag = 0
            counter = 0
            while flag == 0:
                # generate random sphere center coordinates cx,cy,cz
                cx = np.random.choice(a_vol)
                cy = np.random.choice(b_vol)
                cz = np.random.choice(c_vol)
                # generate bounding box around the center coordinate
                # cx+radius is in, cx+radius+1 is out
                cxs_bb = np.arange(cx-radius,cx+radius+1)
                cys_bb = np.arange(cy-radius,cy+radius+1)
                czs_bb = np.arange(cz-radius,cz+radius+1)
                
                # generate a sphere in the bounding box             
                in_sphere_arr = make_sphere(cx,cy,cz,radius,cxs_bb,cys_bb,czs_bb)
                # project the local coordinates of the sphere in the global volume to check for overlap
                xs, ys, zs = calc_global_overlap_coords(in_sphere_arr, cx, cy, cz, radius)  
    
                # introduce infinite boundary conditions
                xs_corr = correct_boundaries(a,xs).astype(int)
                ys_corr = correct_boundaries(b,ys).astype(int)
                zs_corr = correct_boundaries(c,zs).astype(int)
    
                overlap = check_overlap(matrix_vol[xs_corr,ys_corr, zs_corr],matrix_color)
                
                counter += 1
                 
                if overlap == False:
                    matrix_vol[xs_corr,ys_corr,zs_corr] = particle_color
                    flag = 1
                    
                
    ############## SLice ##############################
    number_of_slices = 15
    slices = np.linspace(10,b-10,number_of_slices,dtype=int)             
    
    summary_crack = []
                
    for s in slices:
                    
        matrix_area = matrix_vol[:,s,:]
        
        
        matrix_area_binary_opened = ndimage.binary_opening(matrix_area).astype(matrix_area.dtype)
    
    ########### Indent #################################
        
        center_indent_x = int(a/2)
        center_indent_y = int(b/2)    
        
        d1 = 50
        d2 = 50
        
        crack_width = 3
        crack_length = 300
        
        labeled_matrix = measure.label(matrix_area_binary_opened)
        
        #1st rectangle
        rect_xs1, rect_ys1 = calc_rectangle_coords(center_indent_x, center_indent_y,crack_width,crack_length, angle=0)
        
        particle_content1 = []
        particle_content1 = check_for_particles_in_crack(labeled_matrix, rect_xs1, rect_ys1)
        
        matrix_area_binary_opened = draw_rectangle(matrix_area_binary_opened, rect_xs1, rect_ys1, crack_color)
        
        solid_particles1 = np.sum(np.array(particle_content1)) / (rect_xs1.size * rect_ys1.size)
        
        #2nd rectangle
        
        rect_xs2, rect_ys2 = calc_rectangle_coords(center_indent_x, center_indent_y,crack_width,crack_length, angle=90)
        
        particle_content2 = []
        particle_content2 = check_for_particles_in_crack(labeled_matrix, rect_xs2, rect_ys2)
        
        matrix_area_binary_opened = draw_rectangle(matrix_area_binary_opened, rect_xs2, rect_ys2, crack_color)
        
        solid_particles2 = np.sum(np.array(particle_content2)) / (rect_xs2.size * rect_ys2.size)

        
        
        
        summary_crack.append(np.mean(np.array([solid_particles1, solid_particles2])))
        
    summary_total.append(summary_crack)
    

########### Plot #################################

#means = []
#data = []
#for i in summary_total:
#    means.append(np.median(np.array(i)))
#    
#    data.append(i)
#
#pl.scatter(volume_fraction_of_particles_totals, means)
#
#pl.figure()
#pl.boxplot(data, 1, 'gD')

print("--- %s seconds ---" % (time.time() - start_time))

#import csv
#csvfile = "content_0.3_file1.txt"
#
##Assuming res is a flat list
#with open(csvfile, "w") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    for val in summary_crack:
#        writer.writerow([val])    

##Assuming res is a list of lists
#with open(csvfile, "w") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    writer.writerows(res)


##plot the last distribution matrix
#pl.imshow(matrix_area, cmap='gray')    
#pl.colorbar()


#plot the last distribution matrix
fig = pl.figure()
ax = fig.add_subplot(111)
ax.imshow(matrix_area_binary_opened, cmap='gray')   
indent = patches.Rectangle((a/2 , -d2/2),d1,d2, color='white', alpha=0.3, edgecolor='red') 
#transform = matplotlib.transforms.Affine2D().rotate_deg(45) + ax.transData
#indent.set_transform(transform)
#ax.add_patch(indent)


################ Test ##########################

def make_test_sphere(r):

    radius = r
    cx,cy,cz = 0,0,0
    cxs_bb = np.arange(cx-radius,cx+radius+1)
    cys_bb = np.arange(cy-radius,cy+radius+1)
    czs_bb = np.arange(cz-radius,cz+radius+1)
    sphere = make_sphere(cx,cy,cz,radius,cxs_bb,cys_bb,czs_bb)
    return sphere
    






















