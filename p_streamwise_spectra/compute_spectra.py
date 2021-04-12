import numpy as np

import matplotlib.pyplot as plt

import os



spartan_path    = '/data/cephfs/punim0600/nek/cases_nek/run/pipe/single_phase/retau_180_5M_pipe/'

#%% FUNCS
def read_grid(path):

    f = open(path, "r")
    f = f.read()
    
    f = f.replace('\n','')
    f = f.split(' ')
    
    for i in range(100):
        if f[i] == 'TOTP':
            temp = f[i:i+100]
            temp = list(filter(lambda x: x != '', temp))
            
            TOTP = int(temp[1])
            NUMR = int(temp[2])
            NUMT = int(temp[3])
            NUMZ = int(temp[4])
            
    for i in range(100):
        if f[i] == 'z':
            ind = i + 1
    
    f   = f[ind:]
    f = list(filter(lambda x: x != '', f))
    f = np.array(f,dtype = np.float)
        
    
    
    return f, TOTP, NUMR, NUMT, NUMZ

def read_vel(path):
    f = open(path, "r")
    f = f.read()
    
    f = f.replace('\n',' ')
    f = f.split(' ')
    f = list(filter(lambda x: x != '', f))
    f = np.array(f,dtype = np.float)
    return f

def re_shape_arr(slice_1, N, NUMX, NUMY, NUMZ):
    
    x   = slice_1[: N]
    y   = slice_1[N:2*N]
    z   = slice_1[2*N:3*N]
    
    return x,y,z


field_path = spartan_path + '/spectra_grid'

#%% READ IN FIELD
if 'my_grid' not in locals():
    my_grid, TOTP, NUMR, NUMTHETA, NUMZ      = read_grid(field_path)

    x,y,z                               = re_shape_arr(my_grid, TOTP, NUMR, NUMTHETA, NUMZ)


#%% FIND FILES FOR STUDY
my_fields   = []
times       = []
for file in os.listdir(spartan_path):
    if file.startswith("phi_uu_field"):  
        my_fields.append(file)
        times.append(np.float(file.strip('phi_uu_field')))
        

counter  = 0
for field in my_fields:  
    
    path    = spartan_path + field
    my_u                                 = read_vel(path)
                                     
    
    num_per_slice   = NUMTHETA*NUMZ
    
    
    
    E_arr        = np.zeros((NUMR,int(NUMZ/2)))
    
    nu           = 1/360
    
    for i in range(NUMR):#NUMY
        args     = [i*(num_per_slice), (i+1)*(num_per_slice)]
        
        w_phi    = np.fft.rfft(my_u[args[0]:args[1]].reshape(NUMTHETA,NUMZ))/NUMZ
        
        E        = np.mean(2*np.abs(w_phi)**2,axis=0)
        freqs    = np.fft.rfftfreq(NUMZ, d = 2*np.pi/(NUMZ))[:]
        lambda_f = 1/freqs
        lambda_f = lambda_f[1:]
        k        = 2*np.pi/lambda_f
        E[0]     = 0.5*E[0]
        E        = E[1:]
        E_arr[i,:] = E
    
        lambda_f   = lambda_f / nu
        
    y_plus     = 1/nu - np.unique(y)[np.unique(y) > -1e-12] / nu
    y_plus     = y_plus[::-1]
    
    E_arrf     = np.copy(E_arr)

    radius = np.sqrt(x[:]**2+y[:]**2)
    
    radius = radius[::num_per_slice]
    
    
    rplus = radius * (1/nu)
    
    y_plus = 0.5/nu - rplus
    
    np.savez('./spectra/spectra_streamwise'+ str(times[counter])+ '.npz', k = k, E_arrf = E_arrf, lambda_f = lambda_f, y_plus = y_plus)

    counter = counter + 1
