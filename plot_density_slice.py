import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.default']='regular'
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.cm as cm
from mpi4py import MPI
from sys import exit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

fontpath='/ccs/home/evans/.fonts/Helvetica.ttf'
helvetica=matplotlib.font_manager.FontProperties(size=14,fname=fontpath)

istart = rank+280
iend = rank+280
dnamein='./128/hdf5/raw/'
dnameout='./128/png/'

# some constants
l_s = 3.086e21 # length scale, centimeters in a kiloparsec
m_s = 1.99e33 # mass scale, g in a solar mass
t_s = 3.154e10 # time scale, seconds in a kyr
d_s = m_s / l_s**3 # density scale, M_sun / kpc^3
v_s = l_s / t_s # velocity scale, kpc / kyr
p_s = d_s*v_s**2 # pressure scale, M_sun / kpc kyr^2
G = 6.67259e-8 # in cm^3 g^-1 s^-2
mp = 1.67e-24 # proton mass in grams
G = G / l_s**3 * m_s * t_s**2 # in kpc^3 / M_sun / kyr^2
KB = 1.3806e-16 # boltzmann constant in cm^2 g / s^2 K
M_vir = 1e12
M_d = 6.5e10
M_b = 1e10
M_h = M_vir - M_d #- M_b
R_vir = 261 # MW viral radius in kpc
c_vir = 20 
R_h = R_vir / c_vir # halo scale radius in kpc
R_d = 3.5 # stellar disk scale length in kpc
z_d = 3.5/5.0 # disk scale height in kpc
R_g = 2*R_d # gas disk scale length in kpc
v_to_kmps = l_s/t_s/100000
kmps_to_kpcpkyr = 1.0220122e-6

for i in range(istart,iend+1):
  
  print(i)
  f = h5py.File(dnamein+str(i)+'.h5.0', 'r')
  head = f.attrs
  gamma = head['gamma'][0]
  t = head['t']
  nx = head['dims'][0]
  ny = head['dims'][1]
  nz = head['dims'][2]
  d  = np.array(f['density'])
  #mx = np.array(f['momentum_x'])
  #my = np.array(f['momentum_y'])
  #mz = np.array(f['momentum_z'])
  #E  = np.array(f['Energy'])
  #vx = mx/d
  #vy = my/d
  #vz = mz/d
  #p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0)
  #e  = p/d/(gamma - 1.0)
  #T = (p*0.6*mp / (d*KB)) * (l_s**2 / t_s**2)
  #log_T = np.log10(T)
  log_d = np.log10(d)
  dxy = log_d[:,:,nz/2]
  dxz = log_d[:,ny/2,:]
  print(np.min(log_d), np.max(log_d))
  

  fig = plt.figure(figsize=(4,8), dpi=100)
  a0 = plt.axes([0.,0.5,1.,0.5])
  for child in a0.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 1, 0.1)+400)
  a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a0.imshow(dxy.T, origin='lower', extent=(0, 400, 401, 800))
  a0.autoscale(False)
  a0.text(360, 760, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(440, 280, 320, color='white')
  a0.text(325, 435, '5 kpc', color='white')
  a1 = plt.axes([0.,0.,1.,0.5])
  for child in a1.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a1.set_xticks(400*np.arange(0.1, 1, 0.1))
  a1.set_yticks(400*np.arange(0.1, 1, 0.1))
  a1.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a1.imshow(dxz.T, origin='lower', extent=(0, 400, 0, 400))
  #pretty white border
  a0.axvline(x=0, color='white')
  a1.axvline(x=0, color='white')
  a0.axvline(x=400, color='white')
  a1.axvline(x=400, color='white')
  a1.hlines(400, 0, 400, color='white')
  a1.hlines(0, 0, 400, color='white')
  plt.savefig(dnameout+'d_'+str(i)+'.png', dpi=300)
  plt.close(fig)

