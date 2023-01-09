#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:22:51 2022

@author: liuyiming
"""
#%%
from dolfin import *
import mshr
from ufl import indices
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import time


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


class PeriodicBoundary(SubDomain):
	def __init__(self,xmin,xmax, tolerance=DOLFIN_EPS):

		SubDomain.__init__(self, tolerance)
		self.tolerance = tolerance
		self.xmin = xmin
		self.xmax = xmax
	def inside(self, x, on_boundary):
		TOL = self.tolerance
		
		isInside = (near(x[0], self.xmin, TOL) and on_boundary)

		return isInside

	def map(self, x, y):
		TOL = self.tolerance
		if near(x[0], self.xmax, TOL):
			y[0] = x[0] - (self.xmax-self.xmin)
			y[1] = x[1]
		else:
			y[0]=x[0]
			y[1]=x[1]

q_bds=False
#q_bds=True

processID =  MPI.comm_world.Get_rank()
set_log_level(50) #50, 40, 30, 20, 16, 13, 10

foldername = "Polarized_capacitor_pd/"
geoname="PlateCapacitorDielectrics"
#geoname="free_jet_sharp"
fname = "geo/"
#mesh = Mesh(fname + geoname + '.xml')
omega=10e9
t_end=1.5/omega*2*pi
nt=18
dt=t_end/nt
q_bds=False
#q_bds=True
scale=2
xmin=-0.5*scale
xmax=0.5*scale
ymin=-5*scale
ymax=5*scale
res1=40
res2=400
mesh = RectangleMesh(Point(xmin,ymin), Point(xmax,ymax),res1,res2,"left/right")

plot(mesh)
#%%
r_plate=1.
d_plate=0.5
h_phase=d_plate
l_phase=1.
x_phase=0.
y_phase=0.
x_l=x_phase-l_phase/2
x_r=x_phase+l_phase/2
y_t=y_phase + h_phase/2
y_b=y_phase - h_phase/2


scalar = FiniteElement('P', mesh.ufl_cell(), 1)
vector = VectorElement('DG', mesh.ufl_cell(), 0)

ScalarSpace = FunctionSpace(mesh, scalar,constrained_domain=PeriodicBoundary(xmin,xmax, tolerance=1e-12)) 
VectorSpace = FunctionSpace(mesh, vector,constrained_domain=PeriodicBoundary(xmin,xmax, tolerance=1e-12)) 
ScalarSpace2 = FunctionSpace(mesh, scalar,constrained_domain=PeriodicBoundary(xmin,xmax, tolerance=1e-12)) 



boundaries = "(near(x[1],ymin) && on_boundary)||(near(x[1],ymax) && on_boundary)"
top_plate = "  x[0] <= r && -r<= x[0] && near(x[1], d/2) "
bottom_plate = "x[0] <= r && -r<= x[0] && near(x[1], -d/2) "

#phases between two plates: marked by 1,2,3
phase_1 = "x[0] <= x_l && -r<= x[0] && x[1] <= y_t && y_b <= x[1]"
phase_2 = "x[0] <= x_r && x_l<= x[0] && x[1] <= y_t && y_b <= x[1]"
phase_3 = "x[0] <= r && x_r <= x[0] && x[1] <= y_t && y_b <= x[1]"
facets = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
cells = MeshFunction("size_t", mesh, mesh.geometry().dim() )

#phases out of two plates, marked by 0
cells.set_all(0)

phase1 = CompiledSubDomain(phase_1,r=r_plate,x_l=x_l,y_t=y_t,y_b=y_b)
phase1.mark(cells,1)
phase2 = CompiledSubDomain(phase_2,x_r=x_r,x_l=x_l,y_t=y_t,y_b=y_b)
phase2.mark(cells,2)
phase3 = CompiledSubDomain(phase_3,r=r_plate,x_r=x_r,y_t=y_t,y_b=y_b)
phase3.mark(cells,3)

facets.set_all(0)
boundary = CompiledSubDomain(boundaries,ymin=ymin,ymax=ymax)
boundary.mark(facets, 3)
topPlate = CompiledSubDomain(top_plate,r=r_plate,d=d_plate)
topPlate.mark(facets,1) #top plate
bottomPlate = CompiledSubDomain(bottom_plate,r=r_plate,d=d_plate)
bottomPlate.mark(facets,2) #bottom plate

#plot(SubMesh(mesh,facets,1))
#plot(SubMesh(mesh,facets,2))
file = File(foldername + "subdomains_pb.pvd")
file << facets
#file << cells

n = FacetNormal(mesh)

#interface , area , volume elements
di = Measure( 'dS' , domain=mesh , subdomain_data=facets )
da = Measure( 'ds' , domain=mesh , subdomain_data=facets )
dv = Measure( 'dx' , domain=mesh , subdomain_data=cells )




def complicated_func(t):
	tt=t*omega
	Volt = sin(tt)
	return Volt

Voltage_up = Expression('c',c=complicated_func(0),degree=2)
Voltage_down = Expression('-c',c=complicated_func(0),degree=2)
# Define inflow profile
#%%
#v_in = Expression(('v','0.'), degree=1,v=1./30.)
bc = []
bcE= []
# Define boundary conditions

bc.append( DirichletBC(ScalarSpace, Voltage_up, facets, 1) )
bc.append( DirichletBC(ScalarSpace, Voltage_down, facets, 2) )
bc.append( DirichletBC(ScalarSpace, Constant(0.), facets, 3) )
mark=1


del_phi = TestFunction(ScalarSpace) 
phi = Function(ScalarSpace) #normal of interface calculate from level set function
phi0 = Function(ScalarSpace)
del_E = TestFunction(VectorSpace) 
E_ = Function(VectorSpace) 
E_0 = Function(VectorSpace)
del_q = TestFunction(ScalarSpace2)
q = Function(ScalarSpace2)
mD_=Function(VectorSpace)
file_q = File(foldername+geoname+'_q.pvd')
file_mD = File(foldername+geoname+'mD.pvd')
file_Phi = File(foldername+geoname+'_Phi.pvd')
file_E = File(foldername+geoname+'_E.pvd')
file_eps = File(foldername+geoname+'_eps.pvd')
#%%
V = FunctionSpace(mesh, 'P', 1)



#%%
er=2.03
eps_r0=1.
eps_r1=er
eps_r2=er
eps_r3=er
eps_rr = material_cofficient(mesh,cells,[eps_r0,eps_r1,eps_r2,eps_r3])
	
em=0.008/(omega)
eps_rm0=0
eps_r1m=em
eps_r2m=em
eps_r3m=em
eps_rm = material_cofficient(mesh,cells,[eps_rm0,eps_r1m,eps_r2m,eps_r3m])

file_eps << (eps_rr)
#%%
E = -grad(phi)
E0 = -grad(phi0)

#mD = eps_rr*E
E_dot= (E - E0)/dt
mD_rev = eps_rr*E
mD_diss = eps_rm*E_dot
mD = mD_rev + mD_diss
F= - dot(mD, grad(del_phi)) * (dv(0)+dv(1)+dv(2)+dv(3))
dofs=len(phi.vector())
print('dofs=%d' %dofs)
D_mid = [0]
E_mid = [0]
energy_rev=0
energy_diss=0
t=0
nn=0
while t < t_end:
	t += dt
	nn +=1
	Voltage_up.c=complicated_func(t)
	Voltage_down.c=complicated_func(t)
	solve(F==0, phi, bc)

	if nn > 0 and nn <= 3:
		energy_rev += assemble(dot(mD_rev,E_dot) *(dv(1)+dv(2)+dv(3)))*dt  #phases between two plates: marked by 1,2,3
		energy_diss += assemble(dot(mD_diss,E_dot)*(dv(1)+dv(2)+dv(3)))*dt #phases between two plates: marked by 1,2,3
	if nn > 3 and nn <= 6:
		energy_rev += -assemble(dot(mD_rev,E_dot) *(dv(1)+dv(2)+dv(3)))*dt  #phases between two plates: marked by 1,2,3
		energy_diss += assemble(dot(mD_diss,E_dot)*(dv(1)+dv(2)+dv(3)))*dt #phases between two plates: marked by 1,2,3
	if nn > 6 and nn <= 9:
		energy_rev += assemble(dot(mD_rev,E_dot) *(dv(1)+dv(2)+dv(3)))*dt  #phases between two plates: marked by 1,2,3
		energy_diss += assemble(dot(mD_diss,E_dot)*(dv(1)+dv(2)+dv(3)))*dt #phases between two plates: marked by 1,2,3
	if nn > 9 and nn <= 12:
		energy_rev += -assemble(dot(mD_rev,E_dot) *(dv(1)+dv(2)+dv(3)))*dt  #phases between two plates: marked by 1,2,3
		energy_diss += assemble(dot(mD_diss,E_dot)*(dv(1)+dv(2)+dv(3)))*dt #phases between two plates: marked by 1,2,3

	# energy caculation

#	if nn <= 12:
#		energy_rev += assemble(dot(mD_rev,E_dot) *(dv(1)+dv(2)+dv(3)))*dt  #phases between two plates: marked by 1,2,3
#		energy_diss += assemble(dot(mD_diss,E_dot)*(dv(1)+dv(2)+dv(3)))*dt #phases between two plates: marked by 1,2,3

	E_.assign(project(E,VectorSpace))
	mD_.assign(project(mD,VectorSpace))
	phi0.assign(phi)
	
	if nn == 1 or (nn % 10 == 0 and nn<=10000):
		file_mD << (mD_,t)
		file_Phi << (phi,t)
		file_E << (E_,t)
	if nn <= 13:
		D_mid.append(mD_(Point(0,0))[1])
		E_mid.append(E_(Point(0,0))[1])

Q=energy_rev/energy_diss
print('energy_rev=%s' %energy_rev)
print('energy_diss=%s' %energy_diss)
print('Q=%s' %Q)


#%%plot
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
plt.rcParams.update(config)
fig=plt.figure()
ax = plt.gca()
font1 = {'family' : 'serif', 'serif':'Times New Roman','weight' : 'normal','size': 14}
plt.rc('font',**font1)
plt.xlabel(r'$E_{y}$', fontsize=14)
#plt.title('$\epsilon_r^{\prime \prime}=$ %s' %em,fontsize=14)
plt.title('$e_2=$ %s' %em,fontsize=14)
plt.ylabel(r'$\mathfrak{D}_{y}$',fontsize=14)
#ax.set_aspect('equal', adjustable='box')
ax.set_aspect(1)
#plt.figure(figsize=(5,5))
plt.plot(E_mid,D_mid)
my_x_ticks = np.arange(-6,8,2)
plt.xticks(my_x_ticks)
plt.xlim(-5,5 )
plt.ylim(-5,5 )
plt.savefig(foldername+ 'E_loss_e2=%s.pdf' %em, format='pdf')