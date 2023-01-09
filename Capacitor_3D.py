#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 21:08:36 2022

@author: liuyiming
"""


from dolfin import *
import mshr
from ufl import indices
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import time
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#%%
krylov_params = parameters["krylov_solver"]
#krylov_params["nonzero_initial_guess"] = True
krylov_params["relative_tolerance"] = 1E-7
krylov_params["absolute_tolerance"] = 1E-9
krylov_params["monitor_convergence"] = False
krylov_params["maximum_iterations"] = 500000

def material_cofficient(target_mesh, cells_list, coeffs):
	coeff_func = Function(FunctionSpace(target_mesh, 'DG', 0))
	markers = np.asarray(cells_list.array(), dtype=np.int32)
	#print(markers)
	coeff_func.vector()[:]=np.choose(markers, coeffs)
	return coeff_func


q_bds=False
#q_bds=True

processID =  MPI.comm_world.Get_rank()
set_log_level(50) #50, 40, 30, 20, 16, 13, 10

foldername = "Polarized_capacitor_3D/"
geoname="Capacitor2220_2"
#geoname="free_jet_sharp"
fname = "geo/"
mesh = Mesh(fname + geoname + '.xml')

#mesh = RectangleMesh(Point(-0.5*scalex,-0.5*scaley), Point(0.5*scalex,0.5*scaley),resx,resy,"left/right")

boundaries = "on_boundary"
Air_metal_up = " x[2] <= 1 && -1<= x[2] && x[1] <= 2.5 && -2.5<= x[1] && near(x[0], 2.85) "
Air_metal_in = "x[2] <= 1 && -1<= x[2] && x[1] <= 2.5 && -2.5<= x[1] && near(x[0], -2.85) "
metal_delec = "x[2] <= 1 && -1<= x[2] && x[1] <= 2.5 && -2.5<= x[1] && (near(x[0], -2.25)|| near(x[0], 2.25))"
metal_air_1 = " x[1] <= 2.5 && -2.5<= x[1] && x[0] <= 2.85 && 2.25<= x[0] && (near(x[2], 1) || near(x[2], -1)) "
metal_air_2 = " x[1] <= 2.5 && -2.5<= x[1] && x[0] <= -2.25 && -2.85<= x[0] && (near(x[2], 1) || near(x[2], -1)) "
metal_air_3 = " x[2] <= 1 && -1<= x[2] && x[0] <= 2.85 && 2.25<= x[0] && (near(x[1], -2.5)|| near(x[1], 2.5))"
metal_air_4 = " x[2] <= 1 && -1<= x[2] && x[0] <= -2.25 && -2.85<= x[0] && (near(x[1], -2.5)|| near(x[1], 2.5))"
Metal_up = " x[2] <= 1 && -1<= x[2] && x[1] <= 2.5 && -2.5<= x[1] && x[0] <= 2.85 && 2.25<= x[0] "
Metal_down = " x[2] <= 1 && -1<= x[2] && x[1] <= 2.5 && -2.5<= x[1] && x[0] <= -2.25 && -2.85<= x[0] "
Delec = " x[2] <= 1 && -1<= x[2] && x[1] <= 2.5 && -2.5<= x[1] && x[0] <= 2.25 && -2.25<= x[0] "
facets = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
cells = MeshFunction("size_t", mesh, mesh.geometry().dim() )
cells.set_all(0)

phase1 = CompiledSubDomain(Metal_up)
phase1.mark(cells,1)
phase2 = CompiledSubDomain(Metal_down)
phase2.mark(cells,2)
phase3 = CompiledSubDomain(Delec)
phase3.mark(cells,3)


facets.set_all(0)
boundary = CompiledSubDomain(boundaries)
boundary.mark(facets, 3)
topPlate = CompiledSubDomain(Air_metal_up)
topPlate.mark(facets,1)
bottomPlate = CompiledSubDomain(Air_metal_in)
bottomPlate.mark(facets,2)
metaldelec = CompiledSubDomain(metal_delec)
metaldelec.mark(facets,4)
metalAir1 = CompiledSubDomain(metal_air_1)
metalAir2 = CompiledSubDomain(metal_air_2)
metalAir3 = CompiledSubDomain(metal_air_3)
metalAir4 = CompiledSubDomain(metal_air_4)
metalAir1.mark(facets,5)
metalAir2.mark(facets,5)
metalAir3.mark(facets,5)
metalAir4.mark(facets,5)
#plot(SubMesh(mesh,facets,1))
#plot(SubMesh(mesh,facets,2))
file = File(foldername + "subdomains.pvd")
file << facets
#file << cells
n = FacetNormal(mesh)

#interface , area , volume elements
di = Measure( 'dS' , domain=mesh , subdomain_data=facets )
da = Measure( 'ds' , domain=mesh , subdomain_data=facets )
dv = Measure( 'dx' , domain=mesh , subdomain_data=cells )


scalar = FiniteElement('P', mesh.ufl_cell(), 1)
vector = VectorElement('DG', mesh.ufl_cell(), 0)

ScalarSpace = FunctionSpace(mesh, scalar) 
VectorSpace = FunctionSpace(mesh, vector) 
ScalarSpace2 = FunctionSpace(mesh, scalar) 


Voltage=2.
# Define inflow profile

#%%
#v_in = Expression(('v','0.'), degree=1,v=1./30.)
bc = []
bcE= []
# Define boundary conditions
#bc.append( DirichletBC(Space.sub(1), v_in,facets, 4) )

bc.append( DirichletBC(ScalarSpace, Constant(Voltage/2), facets, 1) )
bc.append( DirichletBC(ScalarSpace, Constant(-Voltage/2), facets, 2) )
bc.append( DirichletBC(ScalarSpace, Constant(0.), facets, 3) )


del_phi = TestFunction(ScalarSpace) 
phi = Function(ScalarSpace,name='$\phi$') #normal of interface calculate from level set function

del_E = TestFunction(VectorSpace) 
E = Function(VectorSpace, name='$E$') #normal of interface calculate from level set function

del_q = TestFunction(ScalarSpace2)
q = Function(ScalarSpace2)

file_q = File(foldername+geoname+'_q.pvd')
file_mD = File(foldername+geoname+'mD.pvd')
file_Phi = File(foldername+geoname+'_Phi.pvd')
file_Phi_metal_up = File(foldername+geoname+'_Phi_metal_up.pvd')
file_Phi_metal_down = File(foldername+geoname+'_Phi_metal_down.pvd')
file_Phi_delec = File(foldername+geoname+'_Phi_delec.pvd')
file_E = File(foldername+geoname+'_E.pvd')
file_eps = File(foldername+geoname+'_eps.pvd')
#%%
V = FunctionSpace(mesh, 'P', 1)

mesh_metal_up = SubMesh(mesh,cells,1)
mesh_metal_down = SubMesh(mesh,cells,2)
mesh_delec = SubMesh(mesh,cells,3)
Space_metal_up = FunctionSpace(mesh_metal_up,'P', 1)
Space_metal_down = FunctionSpace(mesh_metal_down,'P', 1)
Space_delec = FunctionSpace(mesh_delec,'P', 1)
phi_metal_up = Function(Space_metal_up,name='$\phi$')
phi_metal_down = Function(Space_metal_down,name='$\phi$')
phi_delec = Function(Space_delec,name='$\phi$')
#%%
eps_r0=1.
eps_r1=1.
eps_r2=1.
eps_r3=2.1

eps_r = material_cofficient(mesh,cells,[eps_r0,eps_r1,eps_r2,eps_r3])

mD = - eps_r*grad(phi) 
file_eps << (eps_r)
#%%
#dimensionless
F_E = (dot(grad(phi), del_E) + dot(E,del_E)) * dx



F = dot(mD,n) * del_phi * da(3) - dot(mD, grad(del_phi)) * (dv(0)+dv(1)+dv(2)+dv(3)) + dot(mD('+')-mD('-'),n('+'))*del_phi('+')*(di(4)+di(5))


dofs=len(phi.vector())
print('dofs=%d' %dofs)


solve(F==0, phi, bc)
solve(F_E==0, E, bcE)

mD_=Function(VectorSpace)
mD_.assign(project(mD,VectorSpace))

file_mD << (mD_)
file_Phi << (phi)
file_E << (E)
phi_metal_up.assign(project(phi,Space_metal_up))
phi_metal_down.assign(project(phi,Space_metal_down))
phi_delec.assign(project(phi,Space_delec))
file_Phi_metal_up << phi_metal_up
file_Phi_metal_down << phi_metal_down
file_Phi_delec << phi_delec

C1 = 1/Voltage**2*assemble(eps_r*E**2*(dv(1)+dv(2)+dv(3)))
C2 = 1/Voltage**2*assemble(eps_r*E**2*(dv(0)+dv(1)+dv(2)+dv(3)))
print('Capatance = %s' %C1)
print('Capatance = %s' %C2)

#%%plot
Q0=-assemble(dot(mD('+')-mD('-'),n('+'))*di(1))
C3=Q0/Voltage
print('Q0 = %s'%Q0)
print('Capatance = %s' %C3)
