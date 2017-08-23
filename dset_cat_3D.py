import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ns = rank+21
ne = rank+21
n_procs = 16

for n in range(ns, ne+1):

  dname = './m82_out/256_cool/hdf5/'
  f = h5py.File(dname+str(n)+'.h5', 'w')

  file = h5py.File(dname+'raw/'+str(n)+'.h5.0', 'r')
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

  d  = f.create_dataset("density", (nx, ny, nz), chunks=True)
  mx = f.create_dataset("momentum_x", (nx, ny, nz), chunks=True)
  my = f.create_dataset("momentum_y", (nx, ny, nz), chunks=True)
  mz = f.create_dataset("momentum_z", (nx, ny, nz), chunks=True)
  E  = f.create_dataset("Energy", (nx, ny, nz), chunks=True)
  GE  = f.create_dataset("GasEnergy", (nx, ny, nz), chunks=True)

  d[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl]  = file['density']
  mx[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = file['momentum_x']
  my[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = file['momentum_y']
  mz[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = file['momentum_z']
  E[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl]  = file['Energy']
  GE[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = file['GasEnergy']
  
  file.close()

  for i in range(1, n_procs):

    file = h5py.File(dname+'raw/'+str(n)+'.h5.'+str(i), 'r')
    head = file.attrs
    nxl = head['dims_local'][0]
    nyl = head['dims_local'][1]
    nzl = head['dims_local'][2]
    xs = head['offset'][0]
    ys = head['offset'][1]
    zs = head['offset'][2]

    d[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl]  = file['density']
    mx[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = file['momentum_x']
    my[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = file['momentum_y']
    mz[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = file['momentum_z']
    E[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl]  = file['Energy']
    GE[xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = file['GasEnergy']

    file.close()

  f.close()
