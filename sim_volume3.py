# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:58:45 2016
 
@author: Lukas Gartmair
"""
 
import numpy as np
import matplotlib.pyplot as pl
import scipy.special as sps
from sklearn.utils.extmath import cartesian
from scipy import optimize
 
from mpl_toolkits.mplot3d import Axes3D
 
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
 
def calc_corner_coords(vxs,vys):
    check_size = 30
    corners = [(np.min(vxs),np.min(vys)),(np.min(vxs),np.max(vys)),(np.max(vxs),np.min(vys)),(np.max(vxs),np.max(vys))]
     
    cxs = []
    cys = []
    for corner in corners:
        cxs_tmp = np.arange(corner[0]-int(check_size/2),corner[0] + int(check_size/2))
        cys_tmp = np.arange(corner[1]-int(check_size/2),corner[1] + int(check_size/2))
        cxs.append(cxs_tmp)
        cys.append(cys_tmp)
         
    return cxs,cys
     
def check_corner_for_crack(vxs,vys,matrix_area, particle_color, corner_particle_color, corner_check_area_color, indenter_color, successfull_indent_color):
 
    cxs,cys = calc_corner_coords(vxs,vys)     
    cracks = []
    for c in range(4):
        crack = False
        for i in cxs[c]:
            for j in cys[c]:
                if (matrix_area[i,j] == (indenter_color or successfull_indent_color)):
                    pass
                elif (matrix_area[i,j] == particle_color):
                    matrix_area[i,j] = corner_particle_color
                    crack = True
                elif (matrix_area[i,j] == matrix_color):
                    matrix_area[i,j] = corner_check_area_color
        cracks.append(crack)
    return cracks
     
def in_sphere(center_x, center_y, center_z, radius, x, y,z):
    square_dist = ((center_x - x) ** 2 + (center_y - y) ** 2 + (center_z - z) ** 2 )
    return square_dist <= radius ** 2
 
def make_sphere(cx,cy,cz,radius,cxs,cys,czs):
    in_sphere_arr = np.arange(0)
    cxx, cyy, czz = np.meshgrid(cxs,cys,czs)
    in_sphere_arr = np.array(in_sphere(cx,cy,cz,radius,cxx,cyy,czz))
    return in_sphere_arr.astype(int)
     
def calc_overlap_coords(matrix_subset, cx, cy, cz, radius):
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
 
def sieve_particles(diameters,volume_percents):    
    treshold = 100
     
    diameters_new = np.take(diameters,np.where(diameters<treshold)[0])
    volume_percents_new = np.take(volume_percents,np.where(diameters<treshold)[0])
        
    diameters_new1 = np.take(diameters_new,np.where(volume_percents_new!=0)[0])
    volume_percents_new1 = np.take(volume_percents_new,np.where(volume_percents_new!=0)[0])
     
    return diameters_new1, volume_percents_new1
 
################### DATA #####################
particle_sizes = np.array([  1.00000000e-02,   1.10000000e-02,   1.30000000e-02,
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
 
volume_percents = np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.01,  0.06,  0.1 ,  0.15,  0.21,
        0.29,  0.38,  0.49,  0.62,  0.77,  0.96,  1.19,  1.46,  1.78,
        2.16,  2.6 ,  3.08,  3.61,  4.17,  4.72,  5.24,  5.68,  6.  ,
        6.17,  6.16,  5.97,  5.62,  5.13,  4.54,  3.91,  3.28,  2.7 ,
        2.2 ,  1.78,  1.45,  1.19,  1.  ,  0.84,  0.71,  0.59,  0.45,
        0.33,  0.19,  0.06,  0.01,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ])
 
 
#############################
##########  MAIN ############
#############################    
     
#  color settings    
matrix_color = 0
particle_color = 3
successfull_indent_color = 1
indenter_color = 4
corner_check_area_color = 2
corner_particle_color = 5
 
# circle featuress
 
indenter_original_size = 212
 
d1 = indenter_original_size
d2 = indenter_original_size

size = 300 
 
a = size
b = size
c = size 
vol_total = a*b*c
 
vol_fractions = [0.009]
#area_fractions = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.3])
 
cracks_summary = []
hits_summary = []
hits_area_fs = []
actual_afs = []


 
for vol_fraction in vol_fractions:
     
    vol_particles = vol_total*vol_fraction    
     
################ Distribution
######################## ungleichmäßig    
#    
    particle_sizes, volume_percents = sieve_particles(particle_sizes, volume_percents)
     
    numbers = (vol_particles*(volume_percents/100))/((4/3*(particle_sizes/2)**3)*np.pi)
     
    diameters = []
    for x,p in enumerate(particle_sizes):
        for i in range(int(np.round(numbers[x]))):
            diameters.append(p)
    radii = np.array(diameters)/2
     
###########################################
 
####################### gleiche radien
#    
#    radius = 8
#    
#    vol_of_particle = np.pi * (radius**2)
#    
#    number_of_particles = vol_total/vol_particles
#    print(number_of_particles)
#    radii = np.arange(number_of_particles)
#    radii[:] = radius
     
##############################################    
#    
    radii = np.round(radii,decimals=0)
 
    a_out = np.arange(a)
    b_out = np.arange(b)
    c_out = np.arange(c)
     
    #matrix_area = np.zeros((a,b))
    matrix_vol = np.zeros((a,b,c))    
     
    ## place the big ones first, then the small ones this will save a lot of time and money
     
    radii_sorted = np.sort(radii)[::-1]
     
    for i,radius in enumerate(radii_sorted):  
 
        flag = 0
        while flag == 0:
            # first do the square  then select the in the circle
            cx = np.random.choice(a_out)
            cy = np.random.choice(b_out)
            cz = np.random.choice(c_out)
             
            cxs_square = np.arange(cx-radius,cx+radius+1)
            cys_square = np.arange(cy-radius,cy+radius+1)
            czs_square = np.arange(cz-radius,cz+radius+1)
 
            in_sphere_arr = make_sphere(cx,cy,cz,radius,cxs_square,cys_square,czs_square)
            xs, ys, zs = calc_overlap_coords(in_sphere_arr, cx, cy, cz, radius)    
            # infinite boundary conditions
            xs_corr = correct_boundaries(a,xs).astype(int)
            ys_corr = correct_boundaries(b,ys).astype(int)
            zs_corr = correct_boundaries(c,zs).astype(int)
 
            overlap = check_overlap(matrix_vol[xs_corr,ys_corr, zs_corr],matrix_color)
             
            if overlap == False:
                matrix_vol[xs_corr,ys_corr,zs_corr] = particle_color
                flag = 1
                 
#    particles = np.where(matrix_vol == particle_color)        
#    fig = pl.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    nth = 1000
#    ax.plot(particles[0][::nth],particles[1][::nth],particles[2][::nth], linestyle='', color='green', marker='o', markersize=5, alpha=0.2, markeredgecolor = 'none')
#    
# get random slice
 
    c_rnd = np.random.randint(np.min(radii_sorted),c)
    
    # choose the slice / s         
    matrix_areas = []                  
         
    matrix_area1 = matrix_vol[:,b/5,:]
    matrix_area2 = matrix_vol[:,b/2,:]
    matrix_area3 = matrix_vol[:,np.round(b/1.1),:]    
    
    matrix_areas.append(matrix_area1)
    matrix_areas.append(matrix_area2)
    matrix_areas.append(matrix_area3)
#    

    for matrix_area in matrix_areas:
    
        # vickers indents
        hits = []
        cracks_total = []
        number_of_indents = 1
     
        center_indent_x = int(a/2)
        center_indent_y = int(b/2)    
         
        indent_shifter = 0   
         
        for i in range(number_of_indents):
             
            vickers_xs, vickers_ys = calc_indent_coords(d1,d2,indent_shifter, center_indent_x, center_indent_y)
             
            indent = vickers_indent(vickers_xs, vickers_ys, matrix_area,successfull_indent_color, indenter_color, particle_color)
            indent_shifter += d1*2
             
            cracks = check_corner_for_crack(vickers_xs,vickers_ys,matrix_area, particle_color, corner_particle_color, corner_check_area_color, indenter_color, successfull_indent_color)
            hits.append(indent)
            cracks_total.append(cracks)
        cracks_summary.append(np.mean(cracks_total))
        hits_summary.append(np.mean(hits))
        hits_area_fs.append(vol_fraction)
 
############################################
############### Plots
###########################################
 
# indent verteilung
#pl.scatter(hits_area_fs, hits_summary)
#pl.scatter(hits_area_fs, cracks_summary)
 
#plot the last distribution matrix
 
#pl.imshow(matrix_area, cmap='gray')    
#pl.colorbar()

############################################
################################################
######TESTING    
     
import unittest
 
test_radius = 4
test_center_x = 0
test_center_y = 0
test_cxs = np.arange(test_center_x-test_radius,test_center_x+test_radius+1)
test_cys = np.arange(test_center_y-test_radius,test_center_y+test_radius+1)
test_circle_res = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 1, 1, 1, 1, 1, 0, 0],
 [0, 1 ,1, 1, 1 ,1 ,1 ,1 ,0],
 [0 ,1 ,1, 1, 1 ,1 ,1 ,1 ,0],
 [1 ,1 ,1 ,1 ,1, 1, 1, 1, 1],
 [0 ,1 ,1 ,1 ,1, 1 ,1, 1 ,0],
 [0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0],
 [0 ,0 ,1 ,1, 1 ,1 ,1, 0, 0],
 [0 ,0 ,0 ,0, 1, 0 ,0, 0, 0]])
  
test_overlap_matrix = np.zeros((11,11))
test_overlap_matrix2 = np.copy(test_overlap_matrix)
test_overlap_matrix2[5,5] = 1
 
test_circle2 = np.array([[0, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 0]])
test_matrix_to_fill = np.zeros((10,10))
test_radius2 = 3
test_center_x2 = 5
test_center_y2 = 5
test_cxs2 = np.arange(test_center_x2-test_radius2,test_center_x2+test_radius2+1)
test_cys2 = np.arange(test_center_y2-test_radius2,test_center_y2+test_radius2+1)
test_circle2_matrix_filled = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
       [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
 
test_boundary = 20
test_arr = np.arange(10) - 5
test_arr_res = np.array([15, 16, 17, 18, 19,  0,  1,  2,  3,  4])
test_arr2 = np.arange(10) + 15
test_arr_res2 = np.array([15, 16, 17, 18, 19,  0,  1,  2,  3,  4])
 
class MyTest(unittest.TestCase):
 
#    def test_make_circle(self):
#        np.testing.assert_equal(make_circle(test_center_x,test_center_y,test_radius,test_cxs, test_cys),test_circle_res)
#    #def test_fill_matrix(self):
#        #np.testing.assert_equal(fill_matrix(test_matrix_to_fill, test_circle2, test_cxs2, test_cys2,1),test_circle2_matrix_filled)
         
    def test_correct_boundaries(test_cx):
        np.testing.assert_equal(correct_boundaries(test_boundary,test_arr),test_arr_res)
        np.testing.assert_equal(correct_boundaries(test_boundary,test_arr2),test_arr_res2)
 
def main():
    unittest.main()
 
if __name__ == '__main__':
    main()
