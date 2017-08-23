import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ns = rank+121
ne = rank+121
n_procs = 1024

for n in range(ns, ne+1):

  dname = './m82_out/1024/hdf5/'
  f = h5py.File(dname+str(n)+'_proj.h5', 'w')

  file = h5py.File(dname+'raw/'+str(n)+'_proj.h5.0', 'r')
  head = file.attrs
  nx = head['dims'][0]
  ny = head['dims'][1]
  nz = head['dims'][2]
  nxl = head['dims_local'][0]
  nyl = head['dims_local'][1]
  nzl = head['dims_local'][2]
  xs = head['offset'][0]
  ys = head['offset'][1]
  zs = head['offset'][2]
  f.attrs['dims'] = [nx, ny, nz]
  f.attrs['gamma'] = [head['gamma'][0]]
  f.attrs['t'] = [head['t'][0]]
  f.attrs['dt'] = [head['dt'][0]]
  f.attrs['n_step'] = [head['n_step'][0]]
  file.close()


  dxy = np.zeros((nx, ny))
  dxz = np.zeros((nx, nz))
  Txy = np.zeros((nx, ny))
  Txz = np.zeros((nx, nz))

  for i in range(0, n_procs):

    file = h5py.File(dname+'raw/'+str(n)+'_proj.h5.'+str(i), 'r')
    head = file.attrs
    nxl = head['dims_local'][0]
    nyl = head['dims_local'][1]
    nzl = head['dims_local'][2]
    xs = head['offset'][0]
    ys = head['offset'][1]
    zs = head['offset'][2]
    fdxy = np.array(file['d_xy'])
    fdxz = np.array(file['d_xz'])
    fTxy = np.array(file['T_xy'])
    fTxz = np.array(file['T_xz'])

    dxy[xs:xs+nxl,ys:ys+nyl]  += fdxy 
    dxz[xs:xs+nxl,zs:zs+nzl]  += fdxz 
    Txy[xs:xs+nxl,ys:ys+nyl]  += fTxy
    Txz[xs:xs+nxl,zs:zs+nzl]  += fTxz

    file.close()

  d_xy  = f.create_dataset("d_xy", data=dxy)
  d_xz  = f.create_dataset("d_xz", data=dxz)
  T_xy  = f.create_dataset("T_xy", data=Txy)
  T_xz  = f.create_dataset("T_xz", data=Txz)

  f.close()
