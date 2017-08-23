import numpy as np
import matplotlib.pyplot as plt
from array_io import *



fname = "chevalier.txt"
r, M, us, Ps, rhos = read_five_arrays(fname)

fname = "chevalier.dimensionfull.txt"
x, M, u, P, rho, T = read_six_arrays(fname)

mp = 1.6737236e-24
kb = 1.38064852e-16
n = rho / mp

log_r    = np.log10(r)
log_rhos = np.log10(rhos)
log_us   = np.log10(us)
log_Ps   = np.log10(Ps)
log_x    = np.log10(x)
log_M    = np.log10(M)
log_u    = np.log10(u)
log_Pk    = np.log10(P/kb)
log_n    = np.log10(n)
log_T    = np.log10(T)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(wspace=0.3, hspace=0.5, top=0.95)
#fig = plt.figure(figsize=(4,10))
#ax1 = fig.add_axes([0.18, 0.75, 0.75, 0.23])
ax1.plot(x/1000,log_n,color='black')
ax1.set_xlim([0, 5])
ax1.set_ylim([-4.0, 0.0])
ax1.set_xlabel('r [kpc]')
#ax1.axes.xaxis.set_ticklabels([])
#ax1.set_yticks(np.arange(-2.5, 0.0, 0.5))
ax1.set_ylabel('$log_{10}(n)$ $[cm^{-3}]$')
ax1.tick_params(axis='both', direction='in', right='on', top='on')
#ax1.vlines(1, -3.0, 0.0, linestyle='dashed')
#plt.savefig('density.pdf',dpi=300)

#fig = plt.figure(figsize=(4,4))
#ax2 = fig.add_axes([0.18, 0.52, 0.75, 0.23])
ax2.plot(x/1000,log_u,color='black')
ax2.set_xlim([0, 5])
ax2.set_ylim([1, 4.0])
ax2.set_xlabel('r [kpc]')
#ax2.axes.xaxis.set_ticklabels([])
#ax2.set_yticks(np.arange(0.5, 3.5, 0.5))
ax2.set_ylabel('$log_{10}(v)$ [km $s^{-1}$]')
ax2.tick_params(axis='both', direction='in', right='on', top='on')
#ax2.vlines(1, 0.0, 4.0, linestyle='dashed')
#plt.savefig('velocity.pdf',dpi=300)

#fig = plt.figure(figsize=(4,4))
#ax3 = fig.add_axes([0.18, 0.29, 0.75, 0.23])
ax3.plot(x/1000,log_T,color='black')
ax3.set_xlim([0, 5])
ax3.set_ylim([5.0, 9.0])
ax3.set_xlabel('r [kpc]')
#ax3.axes.xaxis.set_ticklabels([])
#ax3.set_yticks(np.arange(6.2, 8.2, 0.4))
ax3.set_ylabel('$log_{10}(T)$ [K]')
ax3.tick_params(axis='both', direction='in', right='on', top='on')
#ax3.vlines(1, 6.0, 8.0, linestyle='dashed')
#plt.savefig('temperature.pdf',dpi=300)

#fig = plt.figure(figsize=(4,4))
#ax4 = fig.add_axes([0.18, 0.06, 0.75, 0.23])
#ax4.plot(x/1000,M,color='black')
#ax4.set_xlim([0, 5])
#ax4.set_ylim([0, 16])
#ax4.set_xlabel('r [kpc]')
#ax4.set_yticks(np.arange(0.0, 10.0, 2.0))
#ax4.set_ylabel('Mach Number')
#ax4.tick_params(axis='both', direction='in', right='on', top='on')
#ax4.vlines(1, 0, 10, linestyle='dashed')
#plt.savefig('mach_number.png',dpi=300)

#fig = plt.figure(figsize=(4,4))
#ax4 = fig.add_axes([0.18, 0.06, 0.75, 0.23])
ax4.plot(x/1000,log_Pk,color='black')
ax4.set_xlim([0, 5])
ax4.set_ylim([2, 8])
ax4.set_xlabel('r [kpc]')
#ax4.set_yticks(np.arange(0.0, 10.0, 2.0))
ax4.set_ylabel('$log_{10}$(P/k) [K $cm^{-3}$]')
ax4.tick_params(axis='both', direction='in', right='on', top='on')
#ax4.vlines(1, 0, 10, linestyle='dashed')
#plt.savefig('pressure.png',dpi=300)


plt.savefig('chevalier.png',dpi=300)


#plt.show()



