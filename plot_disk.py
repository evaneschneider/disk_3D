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

istart = rank+401
iend = rank+401
dnamein='./m82_outflow/1024/hdf5/'
dnameout='./m82_outflow/1024/png/'

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
v_to_kmps = l_s/t_s/100000
kmps_to_kpcpkyr = 1.0220122e-6

for i in range(istart,iend+1):
  
  f = h5py.File(dnamein+str(i)+'.h5', 'r')
  head = f.attrs
  gamma = head['gamma'][0]
  t = head['t']
  nx = head['dims'][0]
  ny = head['dims'][1]
  nz = head['dims'][2]
  d  = np.array(f['density'])
  mx = np.array(f['momentum_x'])
  my = np.array(f['momentum_y'])
  mz = np.array(f['momentum_z'])
  E  = np.array(f['Energy'])
  n  = d*d_s/mp # number density in particles/cc
  vx = mx/d
  vy = my/d
  vz = mz/d
  p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0)
  #e  = p/d/(gamma - 1.0)
  #T = (p*0.6*mp / (d*KB)) * (l_s**2 / t_s**2)
  e = np.array(f['GasEnergy'])
  T = e*p_s*(gamma - 1.0) / (n*KB)
  log_T = np.log10(T)
  zslice = int(nz/2)
  yslice = int(ny/2)
  Txy = log_T[:,:,zslice]
  Txz = log_T[:,yslice,:]

  dx = 10.0 / nx
  dy = 10.0 / ny
  dz = 10.0 / nz

  print '{0:d}, {1:.2e}, {2:.2e}, {3:.2e}, {4:.2e}'.format(i, np.min(d), np.max(d), np.min(T), np.max(T))
  #print '{0:.4e}'.format(np.sum(d*dx*dy*dz))
  
  pdz = np.sum(d*dz, axis=2)
  pdy = np.sum(d*dy, axis=1)
  log_pdz = np.log10(pdz)
  log_pdy = np.log10(pdy)
  pTz = np.sum(T*d*dz, axis=2)/pdz
  pTy = np.sum(T*d*dz, axis=1)/pdy
  log_pTz = np.log10(pTz)
  log_pTy = np.log10(pTy)
  #print(np.min(pdz), np.max(pdz))
  #print(np.min(pdy), np.max(pdy))
  #print(np.min(pTz), np.max(pTz))
  #print(np.min(pTy), np.max(pTy))
  #print(np.min(d), np.max(d))
  #print(np.min(T), np.max(T))
  #pdz_min, pdz_max = 6.25, 7.75
  #pdy_min, pdy_max = 5.90, 8.75
  #pTz_min, pTz_max = 5.40, 6.40
  #pTy_min, pTy_max = 4.80, 6.60
  pdz_min, pdz_max = 6.5, 8.5
  pdy_min, pdy_max = 4.85, 9.75
  pTz_min, pTz_max = 3.80, 6.00
  #pTy_min, pTy_max = 3.30, 6.40
  pTy_min, pTy_max = 3.30, 6.80

  log_pdz = np.clip(log_pdz, pdz_min, pdz_max)
  log_pdy = np.clip(log_pdy, pdy_min, pdy_max)
  log_pTz = np.clip(log_pTz, pTz_min, pTz_max)
  log_pTy = np.clip(log_pTy, pTy_min, pTy_max)

  # make density plot
  fig = plt.figure(figsize=(4,8), dpi=100)
  a0 = plt.axes([0.,0.5,1.,0.5])
  for child in a0.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 1, 0.1)+400)
  a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a0.imshow(log_pdz.T, origin='lower', extent=(0, 400, 401, 800), cmap='bone', vmin=pdz_min, vmax=pdz_max)
  a0.autoscale(False)
  a0.text(360, 760, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(440, 280, 320, color='white')
  a0.text(325, 435, '1 kpc', color='white')
  a1 = plt.axes([0.,0.,1.,0.5])
  for child in a1.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a1.set_xticks(400*np.arange(0.1, 1, 0.1))
  a1.set_yticks(400*np.arange(0.1, 1, 0.1))
  a1.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a1.imshow(log_pdy.T, origin='lower', extent=(0, 400, 0, 400), cmap='bone', vmin=pdy_min, vmax=pdy_max)
  #pretty white border
  a0.axvline(x=0, color='white')
  a1.axvline(x=0, color='white')
  a0.axvline(x=400, color='white')
  a1.axvline(x=400, color='white')
  a1.hlines(400, 0, 400, color='white')
  a1.hlines(0, 0, 400, color='white')
  plt.savefig(dnameout+'d_'+str(i)+'.png', dpi=300)
  plt.close(fig)

  # make temperature plot
  fig = plt.figure(figsize=(4,8), dpi=100)
  a0 = plt.axes([0.,0.5,1.,0.5])
  for child in a0.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 1, 0.1)+400)
  a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a0.imshow(log_pTz.T, origin='lower', extent=(0, 400, 401, 800), cmap='magma', vmin=pTz_min, vmax=pTz_max)
  a0.autoscale(False)
  a0.text(360, 760, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(440, 280, 320, color='white')
  a0.text(325, 435, '1 kpc', color='white')
  a1 = plt.axes([0.,0.,1.,0.5])
  for child in a1.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a1.set_xticks(400*np.arange(0.1, 1, 0.1))
  a1.set_yticks(400*np.arange(0.1, 1, 0.1))
  a1.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a1.imshow(log_pTy.T, origin='lower', extent=(0, 400, 0, 400), cmap='magma', vmin=pTy_min, vmax=pTy_max)
  #pretty white border
  a0.axvline(x=0, color='white')
  a1.axvline(x=0, color='white')
  a0.axvline(x=400, color='white')
  a1.axvline(x=400, color='white')
  a1.hlines(400, 0, 400, color='white')
  a1.hlines(0, 0, 400, color='white')
  plt.savefig(dnameout+'T_'+str(i)+'.png', dpi=300)
  plt.close(fig)

  xwidth = 4
  cbwidth = 0.04
  xlpad = 0.03
  xrpad = 0.03
  cbpad = 0.12
  xlength = 1.0 - xlpad - xrpad - cbwidth - cbpad
  ybpad = 0.02
  ytpad = 0.04
  ywidth = (2*xlength + 2*ybpad + 2*ytpad)*xwidth
  ylength = xlength*(xwidth/ywidth)
  #print(xlength, ywidth, ylength)

  fig = plt.figure(figsize=(xwidth, ywidth), dpi=100)
  a0 = plt.axes([xlpad,0.5+ybpad,xlength,ylength])
#  for child in a0.get_children():
#    if isinstance(child, matplotlib.spines.Spine):
#      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.125, 1, 0.125))
  a0.set_yticks(400*np.arange(0.125, 1, 0.125))
  a0.tick_params(axis='both', which='both', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  image = a0.imshow(Txy.T, origin='lower', extent=(0, 400, 0, 400), cmap='inferno', vmin=3.0, vmax=7.5)
  a0.autoscale(False)
  a0.text(100, 350, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(355, 250, 300, color='white')
  a0.text(305, 350, '1 kpc', color='white')
  #colorbar
  cbaxes = fig.add_axes([xlpad+xlength+xrpad, 0.5+ybpad, cbwidth, ylength])
  cb = plt.colorbar(image, cax = cbaxes, ticks=[np.arange(3.0, 7.6, 0.5)])
  #cb.outline.set_edgecolor('white')
  #cb.solids.set_edgecolor('face')
  #cbaxes.yaxis.set_ticks_position('left')
  #cbaxes.tick_params(axis='y', colors='white')
  #cbaxes.set_ylabel('$log_{10}$(N) [$n_h$ $cm^{-2}$]', color='white')
  #cbaxes.set_ylabel('$log_{10}(N_{H})$ [$cm^{-2}$]')
  #cbaxes.yaxis.set_label_coords(-3.0, 0.5)

  a1 = plt.axes([xlpad,ybpad,xlength,ylength])
#  for child in a1.get_children():
#    if isinstance(child, matplotlib.spines.Spine):
#      child.set_visible(False)  
  a1.set_xticks(400*np.arange(0.125, 1, 0.125))
  a1.set_yticks(400*np.arange(0.125, 1, 0.125))
  a1.tick_params(axis='both', which='both', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  image = a1.imshow(Txz.T, origin='lower', extent=(0, 400, 0, 400), cmap='inferno', vmin=3.0, vmax=7.5)
  a1.autoscale(False)
  cbaxes = fig.add_axes([xlpad+xlength+xrpad, ybpad, cbwidth, ylength])  
  cb = plt.colorbar(image, cax = cbaxes, ticks=[np.arange(3.0, 7.6, 0.5)])
  #pretty white border
#  a0.axvline(x=0, color='white')
#  a1.axvline(x=0, color='white')
#  a0.axvline(x=400, color='white')
#  a1.axvline(x=400, color='white')
#  a1.hlines(200, 0, 400, color='white')
#  a1.hlines(1, 0, 400, color='white')
  plt.savefig(dnameout+'Tslice_'+str(i)+'.png', dpi=300)
  plt.close(fig)
#  plt.show()

