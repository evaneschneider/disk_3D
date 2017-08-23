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

istart = rank+0
iend = rank+0
dnamein='./m82_out/2048/hdf5/'
dnameout='./m82_out/2048/projections/'

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
  
  f = h5py.File(dnamein+str(i)+'_proj.h5', 'r')
  head = f.attrs
  t = head['t']
  nx = head['dims'][0]
  ny = head['dims'][1]
  nz = head['dims'][2]
  pdz  = np.array(f['d_xy'])
  pdy  = np.array(f['d_xz'])
  Tz  = np.array(f['T_xy'])
  Ty  = np.array(f['T_xz'])

  pTz = Tz/pdz
  pTy = Ty/pdy

  pdz_min, pdz_max = 6.5, 8.5
  pdy_min, pdy_max = 4.85, 9.75
  #pTz_min, pTz_max = 3.80, 6.00
  pTz_min, pTz_max = 3.80, 6.80
  pTy_min, pTy_max = 3.30, 6.80

  log_pdz = np.clip(np.log10(pdz), pdz_min, pdz_max)
  log_pdy = np.clip(np.log10(pdy), pdy_min, pdy_max)
  log_pTz = np.clip(np.log10(pTz), pTz_min, pTz_max)
  log_pTy = np.clip(np.log10(pTy), pTy_min, pTy_max)

  # make density plots
  fig = plt.figure(figsize=(4,4), dpi=100)
  a0 = plt.axes([0.,0.,1.,1.])
  for child in a0.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 1, 0.1))
  a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a0.imshow(log_pdz.T, origin='lower', extent=(0, 400, 0, 400), cmap='bone', vmin=pdz_min, vmax=pdz_max)
  a0.autoscale(False)
  a0.text(360, 360, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(40, 280, 320, color='white')
  a0.text(325, 35, '1 kpc', color='white')
  #pretty white border
  a0.axvline(x=0, color='white')
  a0.axvline(x=400, color='white')
  a0.hlines(400, 0, 400, color='white')
  a0.hlines(0, 0, 400, color='white')
  plt.savefig(dnameout+'d_'+str(i)+'_xy.png', dpi=300)
  plt.close(fig)
  fig = plt.figure(figsize=(4,8), dpi=100)
  a0 = plt.axes([0.,0.,1.,1.])
  for child in a0.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 2, 0.1))
  a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a0.imshow(log_pdy.T, origin='lower', extent=(0, 400, 0, 800), cmap='bone', vmin=pdy_min, vmax=pdy_max)
  a0.autoscale(False)
  a0.text(360, 760, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(40, 280, 320, color='white')
  a0.text(325, 35, '1 kpc', color='white')
  #pretty white border
  a0.axvline(x=0, color='white')
  a0.axvline(x=400, color='white')
  a0.hlines(800, 0, 400, color='white')
  a0.hlines(0, 0, 400, color='white')
  plt.savefig(dnameout+'d_'+str(i)+'_xz.png', dpi=300)
  plt.close(fig)

  # make temperature plots
  fig = plt.figure(figsize=(4,4), dpi=100)
  a0 = plt.axes([0.,0.,1.,1.])
  for child in a0.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 1, 0.1))
  a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a0.imshow(log_pTz.T, origin='lower', extent=(0, 400, 0, 400), cmap='magma', vmin=pTz_min, vmax=pTz_max)
  a0.autoscale(False)
  a0.text(360, 360, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(40, 280, 320, color='white')
  a0.text(325, 35, '1 kpc', color='white')
  #pretty white border
  a0.axvline(x=0, color='white')
  a0.axvline(x=400, color='white')
  a0.hlines(400, 0, 400, color='white')
  a0.hlines(0, 0, 400, color='white')
  plt.savefig(dnameout+'T_'+str(i)+'_xy.png', dpi=300)
  plt.close(fig)
  fig = plt.figure(figsize=(4,8), dpi=100)
  a0 = plt.axes([0.,0.,1.,1.])
  for child in a0.get_children():
    if isinstance(child, matplotlib.spines.Spine):
      child.set_visible(False)  
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 2, 0.1))
  a0.tick_params(axis='both', which='both', color='white', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  a0.imshow(log_pTy.T, origin='lower', extent=(0, 400, 0, 800), cmap='magma', vmin=pTy_min, vmax=pTy_max)
  a0.autoscale(False)
  a0.text(360, 760, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(40, 280, 320, color='white')
  a0.text(325, 35, '1 kpc', color='white')
  #pretty white border
  a0.axvline(x=0, color='white')
  a0.axvline(x=400, color='white')
  a0.hlines(800, 0, 400, color='white')
  a0.hlines(0, 0, 400, color='white')
  plt.savefig(dnameout+'T_'+str(i)+'_xz.png', dpi=300)
  plt.close(fig)
