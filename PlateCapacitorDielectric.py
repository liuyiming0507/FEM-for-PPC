#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:05:21 2021

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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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

# choose boundary conditions type
q_bds=False
#q_bds=True

processID =  MPI.comm_world.Get_rank()
set_log_level(50) #50, 40, 30, 20, 16, 13, 10

foldername = "Polarized_capacitor/"
geoname="PlateCapacitorDielectrics"
#geoname="free_jet_sharp"
fname = "geo/"
#mesh = Mesh(fname + geoname + '.xml')
scalex=10
scaley=10
resx=400
resy=400
mesh = RectangleMesh(Point(-0.5*scalex,-0.5*scaley), Point(0.5*scalex,0.5*scaley),resx,resy,"left/right")
r_plate=3
d_plate=3
h_phase=1 
l_phase=1.
x_phase=0.
y_phase=0.
x_l=x_phase-l_phase/2
x_r=x_phase+l_phase/2
y_t=y_phase + h_phase/2
y_b=y_phase - h_phase/2
boundaries = "on_boundary"
top_plate = "  x[0] <= r && -r<= x[0] && near(x[1], d/2) "
bottom_plate = "x[0] <= r && -r<= x[0] && near(x[1], -d/2) "

phase_1 = "x[0] <= x_l && -r<= x[0] && x[1] <= y_t && y_b <= x[1]"
phase_2 = "x[0] <= x_r && x_l<= x[0] && x[1] <= y_t && y_b <= x[1]"
phase_3 = "x[0] <= r && x_r <= x[0] && x[1] <= y_t && y_b <= x[1]"
facets = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
cells = MeshFunction("size_t", mesh, mesh.geometry().dim() )
cells.set_all(0)
# phase 1 - 5 are different regions between two plates. 
phase1 = CompiledSubDomain(phase_1,r=r_plate,x_l=x_l,y_t=y_t,y_b=y_b)
phase1.mark(cells,1)
phase2 = CompiledSubDomain(phase_2,x_r=x_r,x_l=x_l,y_t=y_t,y_b=y_b)
phase2.mark(cells,2)
phase3 = CompiledSubDomain(phase_3,r=r_plate,x_r=x_r,y_t=y_t,y_b=y_b)
phase3.mark(cells,3)

if d_plate != h_phase:
	phase_top = "x[0] <= r && -r<= x[0] && x[1] <= d/2 && y_t <= x[1]"
	phase_bottom = "x[0] <= r && -r<= x[0] && x[1] <= y_b && -d/2 <= x[1]"
	phase4 = CompiledSubDomain(phase_top,r=r_plate,y_t=y_t,d=d_plate)
	phase4.mark(cells,4)
	phase5 = CompiledSubDomain(phase_bottom,r=r_plate,y_b=y_b,d=d_plate)
	phase5.mark(cells,5)

facets.set_all(0)
boundary = CompiledSubDomain(boundaries)
boundary.mark(facets, 3)
topPlate = CompiledSubDomain(top_plate,r=r_plate,d=d_plate)
topPlate.mark(facets,1)
bottomPlate = CompiledSubDomain(bottom_plate,r=r_plate,d=d_plate)
bottomPlate.mark(facets,2)

#plot(SubMesh(mesh,facets,1))
#plot(SubMesh(mesh,facets,2))
file = File(foldername + "subdomains.pvd")
#file << facets
file << cells

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

bc = []
bcE= []
# Define boundary conditions
#bc.append( DirichletBC(Space.sub(1), v_in,facets, 4) )
if q_bds==True:
	bc.append( DirichletBC(ScalarSpace, Constant(0.), facets, 3) )
else:
	bc.append( DirichletBC(ScalarSpace, Constant(Voltage/2), facets, 1) )
	bc.append( DirichletBC(ScalarSpace, Constant(-Voltage/2), facets, 2) )
	bc.append( DirichletBC(ScalarSpace, Constant(0.), facets, 3) )


del_phi = TestFunction(ScalarSpace) 
phi = Function(ScalarSpace) #normal of interface calculate from level set function

del_E = TestFunction(VectorSpace) 
E = Function(VectorSpace) #normal of interface calculate from level set function

del_q = TestFunction(ScalarSpace2)
q = Function(ScalarSpace2)

file_q = File(foldername+geoname+'_q.pvd')
file_mD = File(foldername+geoname+'mD.pvd')
file_Phi = File(foldername+geoname+'_Phi.pvd')
file_E = File(foldername+geoname+'_E.pvd')
file_eps = File(foldername+geoname+'_eps.pvd')
#%%
V = FunctionSpace(mesh, 'P', 1)



#%%
eps_r0=1.
eps_r1=10.
eps_r2=10.
eps_r3=10.
if d_plate != h_phase:
	eps_r4=5.
	eps_r5=2.
	eps_r = material_cofficient(mesh,cells,[eps_r0,eps_r1,eps_r2,eps_r3,eps_r4,eps_r5])
else:
	eps_r = material_cofficient(mesh,cells,[eps_r0,eps_r1,eps_r2,eps_r3])

mD = - eps_r*grad(phi) 
file_eps << (eps_r)
#%%
#dimensionless
F_E = (dot(grad(phi), del_E) + dot(E,del_E)) * dx


if d_plate != h_phase:
	F_ = dot(mD,n) * del_phi * da(3) - dot(mD, grad(del_phi)) * (dv(0)+dv(1)+dv(2)+dv(3)+dv(4)+dv(5)) 
else:
	F_ = dot(mD,n) * del_phi * da(3) - dot(mD, grad(del_phi)) * (dv(0)+dv(1)+dv(2)+dv(3)) 

if q_bds==True:
	Voltage=1.
	C=1.
	q_ =1.* C*Voltage/3.
	initial_q = Expression('((x[0] <= r && -r<= x[0] && near(x[1], d/2) )||( x[0] <= r && -r<= x[0] && near(x[1], -d/2)) )? -2*q_*x[1]/d : 0.0',r=r_plate,d=d_plate,q_=q_, degree=1)
	q.assign(initial_q)
	file_q << (q)
	bd_q = q * del_phi('+') *(di(1)+di(2)) 
	F =  F_ + bd_q
else:
	F=F_ + dot(mD('+')-mD('-'),n('+'))*del_phi('+')*(di(1)+di(2))
	
	
dofs=len(phi.vector())
print('dofs=%d' %dofs)


solve(F==0, phi, bc)
solve(F_E==0, E, bcE)

mD_=Function(VectorSpace)
mD_.assign(project(mD,VectorSpace))

file_mD << (mD_)
file_Phi << (phi)
file_E << (E)


if d_plate != h_phase:
	C1 = 1/Voltage**2*assemble(eps_r*E**2*(dv(1)+dv(2)+dv(3)+dv(4)+dv(5)))
	C2 = 1/Voltage**2*assemble(eps_r*E**2*(dv(0)+dv(1)+dv(2)+dv(3)+dv(4)+dv(5)))
else:
	C1 = 1/Voltage**2*assemble(eps_r*E**2*(dv(1)+dv(2)+dv(3)))
	C2 = 1/Voltage**2*assemble(eps_r*E**2*(dv(0)+dv(1)+dv(2)+dv(3)))
print('Capatance = %s' %C1)
print('Capatance = %s' %C2)

#%%plot
Q0=-assemble(dot(mD('+')-mD('-'),n('+'))*di(1))
C3=Q0/Voltage
print('Q0 = %s'%Q0)
print('Capatance = %s' %C3)
y_=d_plate/2
y_p=y_+0.001
y_n=y_-0.001
nn=0
r=1
dd=0.005
x_=np.arange(-r,r+dd,dd)
q_s= np.zeros(np.size(x_))
for xx in x_:
	q_s[nn]=mD_(Point(xx,y_p))[1]-mD_(Point(xx,y_n))[1]
	nn+=1
plt.plot(x_,q_s,color='red',label='singular interface')
plt.title('Charge density $\ q$ @ Top plate')
plt.legend()
plt.xlabel(r'$y$')
plt.ylabel(r'$\ q$')
my_y_ticks = np.arange(0., 2.5,0.5)
plt.yticks(my_y_ticks)
plt.savefig(foldername+ 'q_@Top plate.pdf', format='pdf')
plt.show()
np.savetxt('q_s.txt', q_s)

Q = np.sum(q_s)*dd
print('Q1 = %s'%Q)

x=0
W=1.
nn=0
r=0.5
y=np.arange(-r+0.01, r, 0.01)
Phi_s= np.zeros(np.size(y))
for yy in y:
	Phi_s[nn] = phi(Point(x,yy))
	nn+=1
plt.plot(y,Phi_s,color='red',label='singular interface')
plt.title('The electric potential @ x=%s' %x)
plt.legend()
plt.xlabel(r'$y$')
plt.ylabel(r'$\phi$')
plt.savefig(foldername+ 'The_electric_potential_singular_@x=%s.pdf' %x, format='pdf')
plt.show()
#np.savetxt('Phi_s_Icharge.txt', Phi_s)
np.savetxt('Phi_s.txt', Phi_s)

x=0
W=1.
nn=0
y=np.arange(-r+0.01, r, 0.01)
E_= np.zeros(np.size(y))
for yy in y:
	E_[nn] = pow(pow(E(Point(x,yy))[0],2)+pow(E(Point(x,yy))[1],2),0.5)
	nn+=1
plt.plot(y,E_,color='red',label='singular interface')
plt.title('The electric field magnitude @ x=%s' %x)
plt.legend()
plt.xlabel(r'$y$')
plt.ylabel(r'$| E |$')

plt.savefig(foldername+ 'The_electric_field_magnitude_singular_@x=%s.pdf' %x, format='pdf')
plt.show()
#np.savetxt('E_s_Icharge.txt', E_)
np.savetxt('E_s.txt', E_)

y_=1.5
W=1.
nn=0
r=1.4
x_=np.arange(-r, r+0.01, 0.01)
Phi_s_x= np.zeros(np.size(x_))
for xx in x_:
	Phi_s_x[nn] = phi(Point(xx,y_))
	nn+=1
plt.plot(x_,Phi_s_x,color='red',label='singular interface')
plt.title('The electric potential @ y=%s' %y_)
plt.legend()
plt.xlabel(r'$y$')
plt.ylabel(r'$\phi$')
my_y_ticks = np.arange(0., 1,0.1)
plt.yticks(my_y_ticks)
plt.savefig(foldername+ 'The_electric_potential_singular_@y=%s.pdf' %y_, format='pdf')
plt.show()
#np.savetxt('Phi_s_Icharge.txt', Phi_s)
np.savetxt('Phi_s_x.txt', Phi_s_x)


y_=0.5
W=1.
nn=0
r=0.5
x_=np.arange(-r, r+0.01, 0.01)
E_x= np.zeros(np.size(x_))
for xx in x_:
	E_x[nn] = pow(pow(E(Point(xx,y_))[0],2)+pow(E(Point(xx,y_))[1],2),0.5)
	nn+=1
plt.plot(x_,E_x,color='red',label='singular interface')
plt.title('The electric field magnitude @ y=%s' %y_)
plt.legend()
plt.xlabel(r'$y$')
plt.ylabel(r'$| E |$')
plt.savefig(foldername+ 'The_electric_field_magnitude_singular_@y=%s.pdf' %y_, format='pdf')
plt.show()
#np.savetxt('E_s_Icharge.txt', E_)
np.savetxt('E_s.txt', E_)

#%%
#Equipotential line
n1, n2 = 100, 100
x_ = np.linspace(-2, 2, n1)
y_ = np.linspace(-2, 2, n2)
nx=0
ny=0

phi_np=np.zeros([np.size(y_),np.size(x_)])
for xx in x_:
	for yy in y_:
		phi_np[ny,nx]=phi(Point(xx,yy))
		ny+=1
	nx+=1
	ny=0

X,Y=np.meshgrid(x_,y_)
font1 = {'family' : 'serif', 'serif':'Times New Roman','weight' : 'normal','size': 15}
plt.rc('font',**font1)
plt.figure(figsize=(6,6))
plt.contour(X,Y,phi_np,levels=np.arange(-0.9,0.9,0.1),colors='steelblue',linestyles='-',linewidths=1.5)
plt.plot([-r_plate,r_plate],[0.5*d_plate,0.5*d_plate],color='black',linewidth=4)
plt.plot([-r_plate,r_plate],[-0.5*d_plate,-0.5*d_plate],color='black',linewidth=4)
plt.xlim(-2.05,2.05 )
plt.ylim(-2.05,2.05 )
ax=plt.gca()
xmajorLocator = MultipleLocator(1) 
xmajorFormatter = FormatStrFormatter('%1.f')  
xminorLocator = MultipleLocator(0.2)  

ymajorLocator = MultipleLocator(1)  
ymajorFormatter = FormatStrFormatter('%1.f')  
yminorLocator = MultipleLocator(0.2)  
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)

ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
'''
my_x_ticks = np.arange(-2.00, 2.001,1)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(-2.00, 2.001,1)
plt.yticks(my_y_ticks)
'''
plt.grid(which='major',alpha=0.5)
#plt.contourf(X,Y,phi_np,8)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(top='on', right='on', which='both')
#plt.savefig(foldername+ 'ElectricPotential.pdf', format='pdf')
plt.show()

#%%
# Grid of x, y points
n1, n2 = 100, 100
x_ = np.linspace(-2, 2, n1)
y_ = np.linspace(-2, 2, n2)
X, Y = np.meshgrid(x_, y_)
# Electric field vector, E=(Ex, Ey), as separate components
Ex, Ey = np.zeros((n2, n1)), np.zeros((n2, n1))
nx=0
ny=0
for xx in x_:
	for yy in y_:
		Ex[ny,nx]=E(Point(xx,yy))[0]
		Ey[ny,nx]=E(Point(xx,yy))[1]
		ny+=1
	nx+=1
	ny=0
font1 = {'family' : 'serif', 'serif':'Times New Roman','weight' : 'normal','size': 15}
plt.rc('font',**font1)
plt.figure(figsize=(6,6))
color = 2 * np.log(np.hypot(Ex, Ey))
#color='steelblue'
plt.streamplot(x_, y_, Ex, Ey, color=color, linewidth=1.5, cmap=plt.cm.inferno, density=2, arrowstyle='->', arrowsize=1.5)
plt.plot([-r_plate,r_plate],[0.5*d_plate,0.5*d_plate],color='black',linewidth=4)
plt.plot([-r_plate,r_plate],[-0.5*d_plate,-0.5*d_plate],color='black',linewidth=4)
plt.xlim(0,2.05 )
plt.ylim(-1,1 )
ax=plt.gca()
xmajorLocator = MultipleLocator(1) 
xmajorFormatter = FormatStrFormatter('%1.f')  
xminorLocator = MultipleLocator(0.2)  

ymajorLocator = MultipleLocator(1)  
ymajorFormatter = FormatStrFormatter('%1.f')  
yminorLocator = MultipleLocator(0.2)  
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)

ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
'''
my_x_ticks = np.arange(-2.00, 2.001,1)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(-2.00, 2.001,1)
plt.yticks(my_y_ticks)
'''
plt.grid(which='major',alpha=0.5)
#plt.contourf(X,Y,phi_np,8)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(top='on', right='on', which='both')
#plt.savefig(foldername+ 'ElectricField.pdf', format='pdf')
plt.show()
