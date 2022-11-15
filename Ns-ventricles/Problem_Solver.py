import mshr
import meshio
import math
import dolfin
from mpi4py import MPI
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import os
import sys
import getopt
import pandas as pd
import scipy.io

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + '/../utilities')


import ParameterFile_handler as prmh
import BCs_handler
import Mesh_handler
import XDMF_handler
import TensorialDiffusion_handler
#import Formulation_Elements as VFE
import Solver_handler
import HDF5_handler
import Common_main

import ErrorComputation_handler


# PROBLEM CONVERGENCE ITERATIONS
def problemconvergence(filename, conv):

	errors = pd.DataFrame(columns = ['Error_L2_u','Error_DG_u','Error_L2_p','Error_DG_p'])

	for it in range(0,conv):
		# Convergence iteration solver
		errors = problemsolver(filename, it, True, errors)
		errors.to_csv("/home/ilaria/Desktop/Navier-Stokes_ventricles/ErrorsDGP4P5.csv")


# PROBLEM SOLVER
def DirichletBoundary(X, param, BoundaryID, time, mesh):

	# Vector initialization
	bc = []

	# Skull Dirichlet BCs Imposition
	period = param['Temporal Discretization']['Problem Periodicity']
		
	if param['Boundary Conditions']['Skull BCs'] == "Dirichlet" :

		# Boundary Condition Extraction Value
		BCsType = param['Boundary Conditions']['Input for Skull BCs']
		BCsValueX = param['Boundary Conditions']['Skull Dirichlet BCs Value (Displacement x-component)']
		
		BCsColumnNameX = param['Boundary Conditions']['File Column Name Skull BCs (x-component)']
		

		# Control of problem dimensionality
		if (mesh.ufl_cell() == triangle):
			BCs = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)

		else:
			BCs = BCs_handler.FindBoundaryConditionValue1D(BCsType, BCsValueX, BCsColumnNameX, time, period)

		# Boundary Condition Imposition
		bc.append(DirichletBC(X, BCs, BoundaryID, 1))
		
		return bc

def problemsolver(filename, iteration = 0, conv = False, errors = False):

	# Import the parameters given the filename
	param = prmh.readprmfile(filename)

	parameters["ghost_mode"] = "shared_facet"

	# Handling of the mesh
	mesh = Mesh_handler.MeshHandler(param, iteration)

	# Importing the Boundary Identifier
	#BoundaryID = BCs_handler.ImportFaceFunction(param, mesh)

	# Computing the mesh dimensionality
	D = mesh.topology().dim()
	# Define function spaces

	# Pressures and Displacement Functional Spaces
	if param['Spatial Discretization']['Method'] == 'DG-FEM':
		V= VectorElement('DG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree velocity']))
		Q= FiniteElement('DG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree pressure']))
    
	elif param['Spatial Discretization']['Method'] == 'CG-FEM':
		V= VectorElement('CG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree velocity']))
		Q= FiniteElement('CG', mesh.ufl_cell(), int(param['Spatial Discretization']['Polynomial Degree pressure']))

	# Mixed FEM Spaces
	W_elem = MixedElement([V,Q])

	# Connecting the FEM element to the mesh discretization
	X = FunctionSpace(mesh, W_elem)

	# Construction of tensorial space
	X9 = TensorFunctionSpace(mesh, "DG", 0)

	# Diffusion tensors definition
	if param['Model Parameters']['Isotropic Diffusion'] == 'No':

		K = Function(X9)
		K = TensorialDiffusion_handler.ImportPermeabilityTensor(param, mesh, K)

	else:
		K = False
	
	if param['Model Parameters']['Steady/Unsteady'] == 'Unsteady':

		# Time step and normal definitions
		dt = Constant(param['Temporal Discretization']['Time Step'])
		T = param['Temporal Discretization']['Final Time']

		n = FacetNormal(mesh)
		
		# Solution functions definition
		x = Function(X)
		u,p= TrialFunctions(X)

		# Test functions definition
		v,q= TestFunctions(X)

		# Previous timestep functions definition
		x_old = Function(X)
		u_old, p_old = x_old.split(deepcopy=True)
		# Measure with boundary distinction definition
		
		#ds_vent, ds_infl = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)
		ds_vent = BCs_handler.MeasuresDefinition(param, mesh, 2)
		ds_infl = BCs_handler.MeasuresDefinition(param, mesh, 1)

		# Time Initialization
		t = 0.0
		
		# Initial Condition Construction
		
		x = InitialConditionConstructor(param, mesh, X, x, p_old, u_old)

		# Output file name definition
		if conv:

				OutputFN = param['Output']['Output XDMF File Name'] + '_Ref' + str(iteration) + '_'
		else:

				OutputFN = param['Output']['Output XDMF File Name']

		# Save the time initial solution
		
		XDMF_handler.NSSolutionSave(OutputFN,x, t, param['Temporal Discretization']['Time Step'])
			
		# Time advancement of the solution
		x_old.assign(x)
		u_old, p_old = split(x_old)

		# Problem Resolution Cicle
		while t < T:
		
		# Temporal advancement
				t += param['Temporal Discretization']['Time Step']

			# Dirichlet Boundary Conditions vector construction
				#bc = VFE.DirichletBoundary(X, param, BoundaryID, t, mesh)

				a, L = VariationalFormulationUnsteadyNSSymGradCERVELLOFullDirichlet(param, u, v,p,q, dt, n, u_old,p_old, K, t, mesh, ds_vent, ds_infl)
			    
			   # Problem Solver Definition
				A = assemble(a)
				b = assemble(L)

				if param['Spatial Discretization']['Method'] == 'CG-FEM':
					[bci.apply(A) for bci in bc]
					[bci.apply(b) for bci in bc]

			# Linear System Resolution
			
				#x = Solver_handler.NSSolver(A, x, b, param)
				x = Solver_handler.NSSolverSenzaMatrici(a, x, L, param)
				
			# Save the solution at time t
				XDMF_handler.NSSolutionSave(OutputFN, x, t, param['Temporal Discretization']['Time Step'])
				
				if (MPI.comm_world.Get_rank() == 0):
					 print("Problem at time {:.6f}".format(t), "solved")
				
			# Time advancement of the solution
				x_old.assign(x)
				u_old, p_old = x_old.split(deepcopy=True)
		ii=plot(u_old)
		plt.colorbar(ii)
		plt.show()
		pp=plot(p_old)
		plt.colorbar(pp)
		plt.show()
		       
	    
	# Error of approximation
		if conv:
		        	errors = ErrorComputation_handler.Navier_Stokes_Errors(param, x, errors, mesh, iteration, t, n)
		if (MPI.comm_world.Get_rank() == 0):
			         print(errors)
		return errors
			
	else:
	
		n = FacetNormal(mesh)
		
		# Solution functions definition
		x = Function(X)
		u,p= TrialFunctions(X)

		# Test functions definition
		v,q= TestFunctions(X)
		a, L = VariationalFormulationDirichSteadySTOKES(param, u, v,p,q, n,  K,  mesh)
			    
			   # Problem Solver Definition
		if conv:

				OutputFN = param['Output']['Output XDMF File Name'] + '_Ref' + str(iteration) + '_'
		else:

				OutputFN = param['Output']['Output XDMF File Name']
		if param['Spatial Discretization']['Method'] == 'CG-FEM':
					[bci.apply(A) for bci in bc]
					[bci.apply(b) for bci in bc]

			# Linear System Resolution
			
		A, bb = assemble_system(a, L)
		U = Function(X)
		solve(A,U.vector(), bb)
				# Get sub-functions
		u, p = U.split()
				
			# Save the solution at time t
		XDMF_handler.NSSolutionSaveSteady(OutputFN, x)
				
		if (MPI.comm_world.Get_rank() == 0):
					print("Problem solved!")
	
	# plot last step
				
		ii=plot(u)
		plt.colorbar(ii)
		plt.show()
		pp=plot(p)
		plt.colorbar(pp)
		plt.show()
			       
		    
		# Error of approximation
		if conv:
				errors = ErrorComputation_handler.Navier_Stokes_ErrorsSteady(param, x, errors, mesh, iteration, n)
			
		       
		if (MPI.comm_world.Get_rank() == 0):
			         print(errors)

		return errors
#########################################################################################################################
#						Variational Formulation Definition					#
#########################################################################################################################
def VariationalFormulationDirichSteadySTOKES(param, u, v, p,q, n, K, mesh):

	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
           
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
	def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        
	def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))

	h = CellDiameter(mesh)
	mu=1.0
	l = param['Spatial Discretization']['Polynomial Degree velocity']
	m=param['Spatial Discretization']['Polynomial Degree pressure']
	fx=param['Model Parameters']['Forcing Terms_1x']
	fy=param['Model Parameters']['Forcing Terms_1y']
	fz=param['Model Parameters']['Forcing Terms_1z']
	gx=param['Model Parameters']['g1 Termsx']
	gy=param['Model Parameters']['g1 Termsy']
	gz=param['Model Parameters']['g1 Termsz']
	if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']))
        	
        	g = Expression((gx,gy), degree=2)
	else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']))
        	
        	g = Expression((gx,gy,gz), degree=2)
        	
	

	# EQUATION  CG
	
	a = (2*mu*inner(sym(grad(u)), grad(v))*dx) - (div(v)*p*dx) + (q*div(u)*dx) + (inner(avg(p), jump(v, n))*dS)\
      + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0)) 
      
	
	L = inner(f, v)*dx  - dot(g, n)*q*ds(0)
	
	# DISCONTINUOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		nu=0.1
 
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gammav'])
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*nu/h('+'),l*l*nu/h('-'))
		sigmav_b=gammav*l*l*nu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	# EQUATION DG
		
		a= a - (mu*inner(avg(grad(v)), tensor_jump(u, n))*dS)\
   - (mu*inner(avg(grad(u)), tensor_jump(v, n))*dS) + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
      + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
      - (inner(avg(q), jump(u, n))*dS) \
      - mu*inner(grad(v), tensor_jump_b(u,n))*ds(0) - mu*inner(grad(u),tensor_jump_b(v,n))*ds(0) \
      + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds (0)
     
		L=L- mu*inner(grad(v), tensor_jump_b(g,n))*ds(0)+ sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0)
	return a, L
   
  
def VariationalFormulationMixSteadyNAVIERSTOKES(param, u, v, p,q, dt, n, u_n,p_n, K, time, mesh, ds_vent, ds_infl):
	def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
           
	def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
	def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        
	def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))

	#time_prev = time-param['Temporal Discretization']['Time Step']
	#theta = param['Temporal Discretization']['Theta-Method Parameter']
	reg = param['Spatial Discretization']['Polynomial Degree']

	h = CellDiameter(mesh)

	
	f=Expression((param['Model Parameters']['Forcing Terms_1']),degree=6, nu=nu)
	g=Expression((param['Model Parameters']['Forcing Terms_2']),degree=6, nu=nu)
	ff=Expression((param['Model Parameters']['Forcing Terms_3']),degree=6, nu=nu)
	
	#f_n=Expression((param['Model Parameters']['Forcing Terms']),degree=6, t=time_prev)

	# EQUATION  CG
	
	a = (nu*inner(grad(u), grad(v))*dx) +inner(grad(u)*u, v)*dx - (div(v)*p*dx) + (q*div(u)*dx) 
    
	
	L= inner(f, v)*dx +dot(ff,v)*ds(1) 
	
	# DISCONTINUOUS GALERKIN TERMS
	if param['Spatial Discretization']['Method'] == 'DG-FEM':

		# Definition of the stabilization parameters
		nu=0.1
 
		gammap=Constant(param['Model Parameters']['gammap'])
		gammav=Constant(param['Model Parameters']['gamma'])
		sigmap=gammap*Min(h('-')/m,h('+')/m)
		sigmav=gammav*Max(l*l*nu/h('+'),l*l*nu/h('-'))
		sigmav_b=gammav*l*l*nu/h
		h_avg = (h('+')*h('-'))*2/(h('-')+h('+'))
	
	# EQUATION DG
		
		a= a - (nu*inner(avg(grad(v)), tensor_jump(u, n))*dS) 
		- (nu*inner(avg(grad(u)), tensor_jump(v, n))*dS)\
                + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS) \
                + (sigmap*dot(jump(p, n), jump(q, n))*dS)\
		+ (inner(avg(p), jump(v, n))*dS) \
		- (inner(avg(q), jump(u, n))*dS) \
                + (dot(v, n)*p*ds(0)) \
                - (dot(u, n)*q*ds(0)) \
                - nu*inner(grad(v), tensor_jump_b(u,n))*ds(0) \
                - nu*inner(grad(u),tensor_jump_b(v,n))*ds(0) \
                + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0) \
                -inner(avg(u),n('+'))* inner(jump(u), avg(v))*dS\
                +0.5*div(u)*inner(u,v)*dx\
                -0.5*dot(jump(u),n('+'))*avg(inner(u,v))*dS\
     
		L=L- nu*inner(grad(v), tensor_jump_b(g,n))*ds(0) \
		+sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0) \
		- dot(g, n)*q*ds(0)
	return a, L


def VariationalFormulationMixUnsteadySTOKES(param, u, v, p, q, dt, n, u_old, p_old, K, time, mesh, ds_vent, ds_infl):

        class top_boundary(SubDomain):
         def inside(self, x, on_boundary):
          return on_boundary and near(x[1], 1)

        class right_boundary(SubDomain):
         def inside(self, x, on_boundary):
          return on_boundary and near(x[0], 1)

        class other_boundary(SubDomain):
         def inside(self, x, on_boundary):
           return near(x[0], 0) or near(x[1], 0)
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
       # def tensor_jump(u, n):
          # return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
       # def tensor_jump_b(u, n):
          # return (outer(u,n)) 
            

        boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)

        top = top_boundary()
        top.mark(boundary_markers, 1)
        right = right_boundary()
        right.mark(boundary_markers, 2)
        other = other_boundary()
        other.mark(boundary_markers, 3)
    
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)


  
        time_prev = time-param['Temporal Discretization']['Time Step']
        theta = Constant(param['Temporal Discretization']['Theta-Method Parameter'])
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
  #gammap=3.1
  #gammav=10.1
        sigmav = 10.1*l**2
        sigmav_b=sigmav
        sigmap = 10.1/l
        h_avg = (h('+') + h('-'))/2
  #sigmap=gammap*(Min(h('-')/m,h('+')/m))
  #sigmav=gammav*Max(l*l*mu/h('+'),l*l*mu/h('-'))
  #sigmav_b=gammav*l*l*mu/h
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gNtx=param['Model Parameters']['g1 Termsx']
        gNty=param['Model Parameters']['g1 Termsy']
        gNtz=param['Model Parameters']['g1 Termsz']
        gNrx=param['Model Parameters']['g2 Termsx']
        gNry=param['Model Parameters']['g2 Termsy']
        gNrz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), mu=mu,t=time)
        	f_old = Expression((fx,fy), degree=int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time_prev, mu=mu)
        	gNt = Expression((gNtx,gNty), degree=2, t=time, mu=mu)
        	gNt_old = Expression((gNtx,gNty), degree=2, t=time_prev, mu=mu)
        	gNr = Expression((gNrx,gNry), degree=2, t=time, mu=mu)
        	gNr_old = Expression((gNrx,gNry), degree=2, t=time_prev, mu=mu)
        else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, mu=mu)
        	f_old = Expression((fx,fy,fz), degree=int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time_prev, mu=mu)
        	gNt = Expression((gNtx,gNty,gNtz), degree=2, t=time, mu=mu)
        	gNt_old = Expression((gNtx,gNty,gNtz), degree=2, t=time_prev, mu=mu)
        	gNr = Expression((gNrx,gNry,gNrz), degree=2, t=time, mu=mu)
        	gNr_old = Expression((gNrx,gNry,gNrz), degree=2, t=time_prev, mu=mu)
        
	# EQUATION  CG    
      
        a= (dot(u, v)/Constant(dt))*dx \
        +(theta)*(mu*inner(grad(u), grad(v)))*dx \
        - div(v)*p*dx\
        + q*div(u)*dx
  
	
        L = (dot(u_old, v)/Constant(dt)) * dx\
+(theta)*(inner(f, v)*dx \
+mu*inner(gNt,v)*ds(1)\
+mu*inner(gNr,v)*ds(2))\
+(1-theta)*(inner(f_old, v)*dx \
+mu*inner(gNt_old,v)*ds(1)\
+mu*inner(gNr_old,v)*ds(2))\
-(1-theta)*(mu*inner(grad(u_old), grad(v)))*dx\
- div(v)*p_old*dx
        
        if param['Spatial Discretization']['Method'] == 'DG-FEM':
        	a= a -mu*inner(avg(grad(v)), tensor_jump(u, n))*dS \
  -mu*inner(tensor_jump(v, n), avg(grad(u)))*dS \
   +sigmav/h_avg*inner(tensor_jump(v, n), tensor_jump(u, n))*dS \
   +sigmap*h_avg*dot(jump(p, n), jump(q, n))*dS \
   + inner(avg(p), jump(v, n))*dS \
   + dot(v, n)*p*ds(3)\
   - mu*inner(grad(v), tensor_jump_b(u,n))*ds(3)\
   - mu*inner(grad(u),tensor_jump_b(v,n))*ds(3)\
   + sigmav_b/h*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(3)\
   - inner(avg(q), jump(u, n))*dS\
   - dot(u, n)*q*ds(3)
  
        	L= L -mu*inner(avg(grad(v)), tensor_jump(u_old, n))*dS \
   -mu*inner(tensor_jump(v, n), avg(grad(u_old)))*dS \
   +sigmav/h_avg*inner(tensor_jump(v, n), tensor_jump(u_old, n))*dS \
   +sigmap*h_avg*dot(jump(p_old, n), jump(q, n))*dS \
   - mu*inner(grad(v), tensor_jump_b(u_old,n))*ds(3)\
   - mu*inner(grad(u_old),tensor_jump_b(v,n))*ds(3)\
   + sigmav_b/h*inner(tensor_jump_b(v,n), tensor_jump_b(u_old,n))*ds(3) \
    + inner(avg(p_old), jump(v, n))*dS \
   + dot(v, n)*p_old*ds(3)
   
   
        return a,L

def VariationalFormulationUnsteadyNAVIERSTOKES(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):

#DEVO PRIMA CAMBIARE F MI SA!!!
	
        class Bottom(SubDomain):
       	 def inside(self, x, on_boundary):
       	  return on_boundary and near(x[1], 0)
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        #def tensor_jump(u, n):
          # return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        #def tensor_jump_b(u, n):
          # return 1/2*(outer(u, n)+ outer(n, u))
        def tensor_jump(u, n):
           return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        def tensor_jump_b(u, n):
           return (outer(u,n)) 
        boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
        bottom = Bottom()
        bottom.mark(boundary_markers, 1)
        
    
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        time_prev = time-param['Temporal Discretization']['Time Step']
        theta = Constant(param['Temporal Discretization']['Theta-Method Parameter'])
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
 
        h_avg = (h('+') + h('-'))/2
        nu=0.1
        gammap=10
        gammav=10
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*nu/h('+'),l*l*nu/h('-'))
        sigmav_b=gammav*l*l*nu/h
 
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), nu=nu,t=time)
        	f_old = Expression((fx,fy), degree=int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time_prev, nu=nu)
        	g= Expression((gx,gy), degree=2, t=time, nu=nu)
        	g_old = Expression((gx,gy), degree=2, t=time_prev, nu=nu)
        	gNb = Expression((gNbx,gNby), degree=2, t=time, nu=nu)
        	gNb_old = Expression((gNbx,gNby), degree=2, t=time_prev, nu=nu)
        else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, nu=nu)
        	f_old = Expression((fx,fy,fz), degree=int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time_prev, nu=nu)
        	g = Expression((gx,gy,gz), degree=2, t=time, nu=nu)
        	g_old = Expression((gx,gy,gz), degree=2, t=time_prev, nu=nu)
        	gNb = Expression((gNbx,gNby,gNbz), degree=2, t=time, nu=nu)
        	gNb_old = Expression((gNbx,gNby,gNz), degree=2, t=time_prev, nu=nu)
        
	# EQUATION  CG    
  
        a= inner(u,v)/Constant(dt)*dx\
        +(nu*inner(grad(u), grad(v))*dx) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (dot(v, n)*p*ds(0)) - (dot(u, n)*q*ds(0)) \
        +inner(grad(u)*u_old, v)*dx\
        +0.5*div(u_old)*inner(u,v)*dx
         
        L= inner(f, v)*dx\
        +inner(u_old,v)/Constant(dt)*dx\
        +dot(gNb,v)*ds(1)\
        -dot(g, n)*q*ds(0)
  
        
        if param['Spatial Discretization']['Method'] == 'DG-FEM':
        	a=  a + (inner(avg(p), jump(v, n))*dS) \
        	-(inner(avg(q), jump(u, n))*dS)\
        	-(nu*inner(avg(grad(v)), tensor_jump(u, n))*dS)\
        	-(nu*inner(avg(grad(u)), tensor_jump(v, n))*dS)\
        	+ (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
        	+ (sigmap*dot(jump(p, n), jump(q, n))*dS) - nu*inner(grad(v), tensor_jump_b(u,n))*ds(0) \
        	- nu*inner(grad(u),tensor_jump_b(v,n))*ds(0)\
        	-inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
        	+ sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0)\
        	-0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS

        	L= L -nu*inner(grad(v), tensor_jump_b(g,n))*ds(0) \
        	+sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0)
   	
        return a, L

def VariationalFormulationUnsteadyNSSymGrad(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):

	
        class Bottom(SubDomain):
       	 def inside(self, x, on_boundary):
       	  return on_boundary and near(x[1], 0)
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n)) 
        boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
        bottom = Bottom()
        bottom.mark(boundary_markers, 1)
        
    
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        time_prev = time-param['Temporal Discretization']['Time Step']
        theta = Constant(param['Temporal Discretization']['Theta-Method Parameter'])
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
 
        h_avg = (h('+') + h('-'))/2
        nu=0.1
        gammap=10
        gammav=10
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*nu/h('+'),l*l*nu/h('-'))
        sigmav_b=gammav*l*l*nu/h
 
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        gNbx=param['Model Parameters']['g2 Termsx']
        gNby=param['Model Parameters']['g2 Termsy']
        gNbz=param['Model Parameters']['g2 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), nu=nu,t=time)
        	f_old = Expression((fx,fy), degree=int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time_prev, nu=nu)
        	g= Expression((gx,gy), degree=2, t=time, nu=nu)
        	g_old = Expression((gx,gy), degree=2, t=time_prev, nu=nu)
        	gNb = Expression((gNbx,gNby), degree=2, t=time, nu=nu)
        	gNb_old = Expression((gNbx,gNby), degree=2, t=time_prev, nu=nu)
        else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, nu=nu)
        	f_old = Expression((fx,fy,fz), degree=int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time_prev, nu=nu)
        	g = Expression((gx,gy,gz), degree=2, t=time, nu=nu)
        	g_old = Expression((gx,gy,gz), degree=2, t=time_prev, nu=nu)
        	gNb = Expression((gNbx,gNby,gNbz), degree=2, t=time, nu=nu)
        	gNb_old = Expression((gNbx,gNby,gNz), degree=2, t=time_prev, nu=nu)
        
	# EQUATION  CG    
  
        a= inner(u,v)/Constant(dt)*dx\
        + (2*nu*inner(sym(grad(u)), sym(grad(v)))*dx) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (dot(v, n)*p*ds(0))\
        - (dot(u, n)*q*ds(0)) \
        +inner(grad(u)*u_old, v)*dx\
        +0.5*div(u_old)*inner(u,v)*dx
         
        L= inner(f, v)*dx\
          +inner(u_old,v)/Constant(dt)*dx\
          +dot(gNb,v)*ds(1)\
          -dot(g, n)*q*ds(0)
  
        
        if param['Spatial Discretization']['Method'] == 'DG-FEM':
        	a=  a + (inner(avg(p), jump(v, n))*dS) \
        	- (inner(avg(q), jump(u, n))*dS)\
                - (2*nu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
                - (2*nu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS)\
                + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
                + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
                - 2*nu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds(0) \
                - 2*nu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds(0)\
                -inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
                + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0)\
                -0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS

        	L= L  -2*nu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds(0) \
        	+sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0)
   	
        return a, L


def VariationalFormulationUnsteadyNSSymGradCERVELLOFullDirichlet(param, u, v, p, q, dt, n, u_old,p_old, K, time, mesh, ds_vent, ds_infl):

	
        def Min(a, b): 
           return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): 
           return (a+b+abs(a-b))/Constant(2)
        
        def tensor_jump(u, n):
           return 1/2*(outer(u('+'),n('+'))+outer(n('+'),u('+')))+1/2*(outer(u('-'),n('-'))+outer(n('-'),u('-')))
        def tensor_jump_b(u, n):
           return 1/2*(outer(u, n)+ outer(n, u))
        #def tensor_jump(u, n):
         #  return  (outer(u('+'),n('+')))+(outer(u('-'),n('-')))
        #def tensor_jump_b(u, n):
         #  return (outer(u,n)) 
         
        boundary_markers = MeshFunction('size_t', mesh, mesh.geometric_dimension()-1, 0)
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        time_prev = time-param['Temporal Discretization']['Time Step']
        theta = Constant(param['Temporal Discretization']['Theta-Method Parameter'])
        l = param['Spatial Discretization']['Polynomial Degree velocity']
        m=param['Spatial Discretization']['Polynomial Degree pressure']

        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        mu=param['Model Parameters']['mu']
 
        h_avg = (h('+') + h('-'))/2
        nu=0.1
        gammap=10
        gammav=10
        sigmap=gammap*(Min(h('-')/m,h('+')/m))
        sigmav=gammav*Max(l*l*nu/h('+'),l*l*nu/h('-'))
        sigmav_b=gammav*l*l*nu/h
 
  
        fx=param['Model Parameters']['Forcing Terms_1x']
        fy=param['Model Parameters']['Forcing Terms_1y']
        fz=param['Model Parameters']['Forcing Terms_1z']
        gx=param['Model Parameters']['g1 Termsx']
        gy=param['Model Parameters']['g1 Termsy']
        gz=param['Model Parameters']['g1 Termsz']
        if (mesh.ufl_cell()==triangle):
        	f = Expression((fx,fy), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), nu=nu,t=time)
        	f_old = Expression((fx,fy), degree=int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time_prev, nu=nu)
        	g= Expression((gx,gy), degree=2, t=time, nu=nu)
        	g_old = Expression((gx,gy), degree=2, t=time_prev, nu=nu)
        	
        else:
        	f = Expression((fx,fy,fz), degree = int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time, nu=nu)
        	f_old = Expression((fx,fy,fz), degree=int(param['Spatial Discretization']['Polynomial Degree velocity']), t=time_prev, nu=nu)
        	g = Expression((gx,gy,gz), degree=2, t=time, nu=nu)
        	g_old = Expression((gx,gy,gz), degree=2, t=time_prev, nu=nu)
        	
	# EQUATION  CG    
  
        a= inner(u,v)/Constant(dt)*dx\
        + (2*nu*inner(sym(grad(u)), sym(grad(v)))*dx) \
        - (div(v)*p*dx) + (q*div(u)*dx) \
        + (dot(v, n)*p*ds(0))\
        - (dot(u, n)*q*ds(0)) \
        +inner(grad(u)*u_old, v)*dx\
        +0.5*div(u_old)*inner(u,v)*dx
         
        L= inner(f, v)*dx\
          +inner(u_old,v)/Constant(dt)*dx\
          -dot(g, n)*q*ds(0)
  
        
        if param['Spatial Discretization']['Method'] == 'DG-FEM':
        	a=  a + (inner(avg(p), jump(v, n))*dS) \
        	- (inner(avg(q), jump(u, n))*dS)\
                - (2*nu*inner(avg(sym(grad(v))), tensor_jump(u, n))*dS) \
                - (2*nu*inner(avg(sym(grad(u))), tensor_jump(v, n))*dS)\
                + (sigmav*inner(tensor_jump(v, n), tensor_jump(u, n))*dS)\
                + (sigmap*dot(jump(p, n), jump(q, n))*dS) \
                - 2*nu*inner(sym(grad(v)), tensor_jump_b(u,n))*ds(0) \
                - 2*nu*inner(sym(grad(u)),tensor_jump_b(v,n))*ds(0)\
                -inner(avg(u_old),n('+'))* inner(jump(u), avg(v))*dS\
                + sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(u,n))*ds(0)\
                -0.5*dot(jump(u_old),n('+'))*avg(inner(u,v))*dS

        	L= L  -2*nu*inner(sym(grad(v)), tensor_jump_b(g,n))*ds(0) \
        	+sigmav_b*inner(tensor_jump_b(v,n), tensor_jump_b(g,n))*ds(0)
   	
        return a, L


##############################################################################################################################
#				Constructor of Initial Condition from file or constant values 				     #
##############################################################################################################################

def InitialConditionConstructor(param, mesh, X, x, p_old, u_old):

	# Solution Initialization
	p0 = param['Model Parameters']['Initial Condition (Pressure)']
	
	u0 = param['Model Parameters']['Initial Condition (Velocity)']

	if (mesh.ufl_cell()==triangle):
		x0 = Constant((p0, u0[0], u0[1]))

	else:
		x0 = Constant((p0, u0[0], u0[1], u0[2]))

	x = interpolate(x0, X)

	# Initial Condition Importing from Files
	if param['Model Parameters']['Initial Condition from File (Pressure)'] == 'Yes':

		p_old = HDF5.ImportICfromFile(param['Model Parameters']['Initial Condition File Name'], mesh, p__old,param['Model Parameters']['Name of IC Function in File'])
		assign(x.sub(0), p_old)

	return x


######################################################################
#				Main 				     #
######################################################################

if __name__ == "__main__":

	Common_main.main(sys.argv[1:], cwd, '/../physics')

	if (MPI.comm_world.Get_rank() == 0):
		print("Problem Solved!")

