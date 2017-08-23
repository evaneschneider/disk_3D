import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.default']='regular'
#matplotlib.rcParams['axes.prop_cycle']=cycler(color='bgrcmyk')
import matplotlib.pyplot as plt
from array_io import *
from matplotlib.colors import hsv_to_rgb
import matplotlib.cm as cm
import matplotlib.lines as mlines
from mpi4py import MPI
from sys import exit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

fontpath='/ccs/home/evans/.fonts/Helvetica.ttf'
helvetica=matplotlib.font_manager.FontProperties(size=14,fname=fontpath)

#istart = rank
#iend = rank
istart = 5
iend = 25
dnamein='./m82_out/512/hdf5/'
dnameout='./m82_out/512/'

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
mu = 0.6

# read in exact solution
fname = "chevalier.dimensionfull.txt"
xc, Mc, uc, Pc, rhoc, Tc = read_six_arrays(fname)

nc = rhoc / (mu*mp)
log_xc    = np.log10(xc)
log_Mc   = np.log10(Mc)
log_uc    = np.log10(uc)
log_Pkc    = np.log10(Pc/KB)
log_nc    = np.log10(nc)
log_Tc    = np.log10(Tc)


# set up figure axis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(wspace=0.3, hspace=0.4)

for i in range(istart,iend+1,10):
  
  print(i)
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
  e  = np.array(f['GasEnergy'])
  n  = d*d_s/(mu*mp)
  vx = mx/d
  vy = my/d
  vz = mz/d
  p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0)
  #e  = p/d/(gamma - 1.0)
  #T = (p*0.6*mp / (d*KB)) * (l_s**2 / t_s**2)
  T  = e*p_s*(gamma-1.0)/(n*KB)
  log_T = np.log10(T)
  print('{0:.2e}, {1:.2e}'.format(np.min(d), np.max(d)))
  print('{0:.2e}, {1:.2e}'.format(np.min(T), np.max(T)))

  #set up grid
  xmin, xmax = -5,5
  ymin, ymax = -5,5
  zmin, zmax = -10,10
  dx = 10 / nx
  dy = 10 / ny
  dz = 20 / nz
  x = np.linspace(xmin+0.5*dx,xmax-0.5*dx,nx)
  y = np.linspace(ymin+0.5*dy,ymax-0.5*dy,ny)
  z = np.linspace(zmin+0.5*dz,zmax-0.5*dz,nz)
  r = np.sqrt(x**2 + y**2)
  x_pos, y_pos = np.meshgrid(x, y, indexing='ij')
  r_pos = np.sqrt(x_pos**2 + y_pos**2)
  xslice = int(nx/2)
  yslice = int(ny/2)
  zslice = int(nz/2)

  nplot = n[xslice, yslice, :]
  vplot = np.abs(vz[xslice, yslice, :])*v_to_kmps
  Tplot = T[xslice, yslice, :]
  Pplot = p[xslice, yslice, :]*p_s/KB

  if (i == 5):
    cplot = 'C0'
  if (i == 15): 
    cplot = 'C1'
  if (i == 25): 
    cplot = 'C2'

  # plot density profile
  ax1.plot(z, np.log10(nplot), color=cplot)
  
  # plot velocity profile
  ax2.plot(z, np.log10(vplot), color=cplot)

  # plot temperature profile
  ax3.plot(z, np.log10(Tplot), color=cplot)

  # plot pressure profile
  ax4.plot(z, np.log10(Pplot), color=cplot)

# plot exact solution
ax1.plot(xc/1000,log_nc,color='k')
ax2.plot(xc/1000,log_uc,color='k')
ax3.plot(xc/1000,log_Tc,color='k')
ax4.plot(xc/1000,log_Pkc,color='k')


ax1.set_xlim([0, 5])  
ax1.set_ylim([-4, 2])  
ax1.set_xlabel('r [kpc]')
ax1.set_ylabel('$log_{10}(n)$ $[cm^{-3}]$')
ax1.tick_params(axis='both', direction='in', right='on', top='on')

ax2.set_xlim([0, 5])  
ax2.set_ylim([1, 4])
ax2.set_xlabel('r [kpc]')
ax2.set_ylabel('$log_{10}(v)$ [km $s^{-1}$]')  
ax2.tick_params(axis='both', direction='in', right='on', top='on')

ax3.set_xlim([0, 5])  
ax3.set_ylim([4, 9])
ax3.set_xlabel('r [kpc]')
ax3.set_ylabel('$log_{10}(T)$ [K]')
ax3.tick_params(axis='both', direction='in', right='on', top='on')

ax4.set_xlim([0, 5])  
ax4.set_ylim([2, 8])
ax4.set_xlabel('r [kpc]')
ax4.set_ylabel('$log_{10}(P/k)$ [K $cm^{-3}$]')
ax4.tick_params(axis='both', direction='in', right='on', top='on')

# legend
line0 = mlines.Line2D([], [], color='C0', label='5 Myr')
line1 = mlines.Line2D([], [], color='C1', label='15 Myr')
line2 = mlines.Line2D([], [], color='C2', label='25 Myr')
line3 = mlines.Line2D([], [], color='k', label='exact solution')
ax1.legend(handles=[line0, line1, line2, line3])

#plt.show()
plt.savefig(dnameout+'profiles_z.png', dpi=300)
plt.close(fig)
