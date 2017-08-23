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

istart = rank+41
iend = rank+41
dnamein='./m82_out/256_cool/hdf5/'
dnameout='./m82_out/256_cool/slices/'

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
  mu = 0.6
  t = head['t']
  nx = head['dims'][0]
  ny = head['dims'][1]
  nz = head['dims'][2]
  d  = np.array(f['density'])
  mx = np.array(f['momentum_x'])
  my = np.array(f['momentum_y'])
  mz = np.array(f['momentum_z'])
  E  = np.array(f['Energy'])
  ge = np.array(f['GasEnergy'])
  n  = d*d_s/(mu*mp)
  vx = mx/d
  vy = my/d
  vz = mz/d
  #p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0)
  #e  = p/d/(gamma - 1.0)
  #T = (p*0.6*mp / (d*KB)) * (l_s**2 / t_s**2)
  T = ge*p_s*(gamma-1.0)/(n*KB)
  log_n = np.log10(n)
  log_T = np.log10(T)
  zslice = int(nz/2)
  yslice = int(ny/2)
  nxy = log_n[:,:,zslice]
  nxz = log_n[:,yslice,:]
  Txy = log_T[:,:,zslice]
  Txz = log_T[:,yslice,:]
  print i, np.min(n), np.max(n), np.min(T), np.max(T)


  xwidth = 4
  cbwidth = 0.04
  xlpad = 0.03
  xrpad = 0.03
  cbpad = 0.15
  xlength = 1.0 - xlpad - xrpad - cbwidth - cbpad
  ybpad = 0.02
  ytpad = 0.04
  ywidth = (2*xlength + ybpad + ytpad)*xwidth
  ylength = xlength*(2*xwidth/ywidth)
  #print(xlength, ywidth, ylength)

  #xz plots
  #density 
  fig = plt.figure(figsize=(xwidth, ywidth), dpi=100)
  a0 = plt.axes([xlpad,ybpad,xlength,ylength])
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 2, 0.1))
  a0.tick_params(axis='both', which='both', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  image = a0.imshow(nxz.T, origin='lower', extent=(0, 400, 0, 800), cmap='viridis', vmin=-4.0, vmax=2.0)
  a0.autoscale(False)
  a0.text(100, 750, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(755, 280, 320, color='white')
  a0.text(325, 750, '1 kpc', color='white')
  #colorbar
  cbaxes = fig.add_axes([xlpad+xlength+xrpad, ybpad, cbwidth, ylength])
  cbaxes.tick_params(axis='y', which='both', length=4, direction='in')  
  cb = plt.colorbar(image, cax = cbaxes, ticks=[np.arange(-4.0, 2.1, 0.5)])
  cb.set_label("n [$n_h$ $cm^{-3}]$")
  plt.savefig(dnameout+'dslice_'+str(i)+'_xz.png', dpi=300)
  plt.close(fig)
  
  #temperature
  fig = plt.figure(figsize=(xwidth, ywidth), dpi=100)
  a0 = plt.axes([xlpad,ybpad,xlength,ylength])
  a0.set_xticks(400*np.arange(0.1, 1, 0.1))
  a0.set_yticks(400*np.arange(0.1, 2, 0.1))
  a0.tick_params(axis='both', which='both', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  image = a0.imshow(Txz.T, origin='lower', extent=(0, 400, 0, 800), cmap='inferno', vmin=3.0, vmax=7.5)
  a0.autoscale(False)
  a0.text(100, 750, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a0.hlines(755, 280, 320, color='white')
  a0.text(325, 750, '1 kpc', color='white')
  #colorbar
  cbaxes = fig.add_axes([xlpad+xlength+xrpad, ybpad, cbwidth, ylength])
  cbaxes.tick_params(axis='y', which='both', length=4, direction='in')  
  cb = plt.colorbar(image, cax = cbaxes, ticks=[np.arange(3.0, 7.6, 0.5)])
  cb.set_label("T [K]")
  plt.savefig(dnameout+'Tslice_'+str(i)+'_xz.png', dpi=300)
  plt.close(fig)

  #xy plots
  xwidth = 4
  cbwidth = 0.04
  xlpad = 0.03
  xrpad = 0.03
  cbpad = 0.15
  xlength = 1.0 - xlpad - xrpad - cbwidth - cbpad
  ybpad = 0.04
  ytpad = 0.03
  ywidth = (xlength + ybpad + ytpad)*xwidth
  ylength = xlength*(xwidth/ywidth)
  #density
  fig = plt.figure(figsize=(xwidth, ywidth), dpi=100)
  a1 = plt.axes([xlpad,ybpad,xlength,ylength])
  a1.set_xticks(400*np.arange(0.1, 1, 0.1))
  a1.set_yticks(400*np.arange(0.1, 1, 0.1))
  a1.tick_params(axis='both', which='both', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  image = a1.imshow(nxy.T, origin='lower', extent=(0, 400, 0, 400), cmap='viridis', vmin=-4.0, vmax=2.0)
  a1.autoscale(False)
  a1.text(100, 350, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a1.hlines(355, 280, 320, color='white')
  a1.text(325, 350, '1 kpc', color='white')
  cbaxes = fig.add_axes([xlpad+xlength+xrpad, ybpad, cbwidth, ylength])  
  cbaxes.tick_params(axis='y', which='both', length=4, direction='in')  
  cb = plt.colorbar(image, cax = cbaxes, ticks=[np.arange(-4.0, 2.1, 0.5)])
  cb.set_label("n [$n_h$ $cm^{-3}]$")
  plt.savefig(dnameout+'dslice_'+str(i)+'_xy.png', dpi=300)
  plt.close(fig)

  # temperature
  fig = plt.figure(figsize=(xwidth, ywidth), dpi=100)
  a1 = plt.axes([xlpad,ybpad,xlength,ylength])
  a1.set_xticks(400*np.arange(0.1, 1, 0.1))
  a1.set_yticks(400*np.arange(0.1, 1, 0.1))
  a1.tick_params(axis='both', which='both', length=5, direction='in', top='on', right='on', labelleft='off', labelbottom='off')  
  image = a1.imshow(Txy.T, origin='lower', extent=(0, 400, 0, 400), cmap='inferno', vmin=3.0, vmax=7.5)
  a1.autoscale(False)
  a1.text(100, 350, str(int(t/1000))+' Myr', color='white', horizontalalignment='right')
  a1.hlines(355, 280, 320, color='white')
  a1.text(325, 350, '1 kpc', color='white')
  cbaxes = fig.add_axes([xlpad+xlength+xrpad, ybpad, cbwidth, ylength])  
  cbaxes.tick_params(axis='y', which='both', length=4, direction='in')  
  cb = plt.colorbar(image, cax = cbaxes, ticks=[np.arange(3.0, 7.6, 0.5)])
  cb.set_label("T [K]")
  plt.savefig(dnameout+'Tslice_'+str(i)+'_xy.png', dpi=300)
  plt.close(fig)

