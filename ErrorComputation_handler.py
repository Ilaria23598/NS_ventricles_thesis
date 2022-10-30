import mshr
import dolfin
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import sys
import pandas as pd


############################################################################################################
# Control the input parameters about the type of mesh to be used and call the generation/import procedures #
############################################################################################################

def Navier_Stokes_Errors(param, c, errors, mesh, iteration, T, n):

	# Importing the exact solutions
	ux=param['Convergence Test']['Exact Solution Velocity x']
	uy=param['Convergence Test']['Exact Solution Velocity y']
	u_ex = Expression((ux,uy), degree=6, t=T)
	p_ex = Expression((param['Convergence Test']['Exact Solution Pressure']), degree=6, t=T)

	# Computing the errors il L2-norm
	        
	Error_L2_u = errornorm(u_ex,u,'L2')
	Error_L2_p = errornorm(p_ex,p,'L2')

	# Computing the errors in H1-norm
	Error_H1_u = errornorm(u_ex,u,'H10')

	if param['Spatial Discretization']['Method'] == "DG-FEM":

		deg = param['Spatial Discretization']['Polynomial Degree']
		h = CellDiameter(mesh)
		sigmav = param['Model Parameters']['sigmav']
		sigmap = param['Model Parameters']['sigmap']

		h_avg = (2*h('+')*h('-'))/(h('+')+h('-'))
		
	# Computing the errors in DG-norm
		
		ErrDGp= sqrt(assemble((p-p_ex)**2*dx)\
		        +assemble(sigmap*dot(jump(p-p_ex, n), jump(p-p_ex, n))*dS))
		        
		ErrDGu=sqrt(mu*errornorm(a, b, 'H10')*errornorm(a, b, 'H10')\
			+sigmav*assemble(inner(tensor_jump(u-u_ex,n),tensor_jump(u-u_ex,n))*dS)\
		+sigmav*assemble(inner(tensor_jump_b(u-u_ex,n),tensor_jump_b(u-u_ex,n))*ds(1)))

	errorsnew = pd.DataFrame({'Error_L2_u': Error_L2,'Error_DG_u': ErrDGu, 'Error_H1' : Error_H1_u,'Error_L2_p': Error_L2_p,'Error_DG_u': ErrDGp}, index=[iteration])

	if iteration == 0:
		errors = errorsnew

	else:
		errors = pd.concat([errors,errorsnew])

	return errors
	 

def FisherKolm_Errors(param, c, errors, mesh, iteration, T, n):

	# Importing the exact solutions
	
	c_ex = Expression((param['Convergence Test']['Exact Solution']), degree=6, t=T)

	# Computing the errors il L2-norm
	X=FunctionSpace(mesh,'DG',1)
	cc=Function(X)
	cc=project(exp(c),X)
	if param['Model Parameters']['c or l'] == "l":
	        c=cc
	        
	Error_L2 = errornorm(c_ex,c,'L2')

	# Computing the errors in H1-norm
	Error_H1 = errornorm(c_ex,c,'H10')

	if param['Spatial Discretization']['Method'] == "DG-FEM":

		deg = param['Spatial Discretization']['Polynomial Degree']
		h = CellDiameter(mesh)
		gamma = param['Model Parameters']['gamma']

		h_avg = (2*h('+')*h('-'))/(h('+')+h('-'))
		d = param['Model Parameters']['Diffusion']

	# Computing the errors in DG-norm

		#I_u = ((deg+1)*(deg+1)*gamma/h*(u-u_ex)*(u-u_ex)*ds) + ((deg+1)*(deg+1)*gamma/h_avg*dot(jump(u-u_ex,n),jump(u-u_ex,n))*dS)
		
		I_u = (deg*deg*gamma/h*(c-c_ex)*(c-c_ex)*ds) + (deg*deg*gamma/h_avg*dot(jump(c-c_ex,n),jump(c-c_ex,n))*dS)
		
		#Error_H1_DG = sqrt(assemble(I_u))+ Error_H1_u
		Error_H1_DG = sqrt(assemble(I_u)+ d*Error_H1*Error_H1)

	errorsnew = pd.DataFrame({'Error_L2': Error_L2,'Error_DG': Error_H1_DG, 'Error_H1' : Error_H1}, index=[iteration])

	if iteration == 0:
		errors = errorsnew

	else:
		errors = pd.concat([errors,errorsnew])

	return errors