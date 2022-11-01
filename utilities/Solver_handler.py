import mshr
import dolfin
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import sys
import pandas as pd
import time

############################################################################################################
# Control the input parameters about the type of mesh to be used and call the generation/import procedures #
############################################################################################################


def NSSolver(A, x, b, param, P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	         
                 Ainv = LUSolver(A)
		 
                 Ainv.solve(x.vector(), b)
	

	return x

def NSSolverSenzaMatrici(a, x, L, param, P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	         
                 solve(a==L, x,solver_parameters={'linear_solver' : 'mumps'})
                 
	

	return x
	
	
	
def LinearSolver(a, x, L, param, P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	
		solve(a==L,x)
		#solve(a==L,x,solver_parameters={'linear_solver' : 'mumps'})
	

	elif param["Linear Solver"]["Type of Solver"] == "Iterative Solver":

		soltype = param["Linear Solver"]["Iterative Solver"]
		precon = param["Linear Solver"]["Preconditioner"]
		A=assemble(a)
		b=assemble(L)

		solver = PETScKrylovSolver(soltype, precon)

		if P == False:
			solver.set_operator(A)

		else:
			solver.set_operators(A,P)

		solver.parameters["relative_tolerance"] = 1e-8
		solver.parameters["absolute_tolerance"] = 1e-8
		solver.parameters["nonzero_initial_guess"] = True
		solver.parameters["monitor_convergence"] = False
		solver.parameters["report"] = True
		solver.parameters["maximum_iterations"] = 100000

		solver.solve(x.vector(), b)

	return x

       

		
	
def NonLinearSolver(a, x, l, L, param, P = False):

	if param["Linear Solver"]["Type of Solver"] == "Default":
	
		a=a-L
		DF=derivative(a,l)
		
		
		x=solve(a==0,l,J=DF,solver_parameters = {"newton_solver":{'maximum_iterations':100000,"linear_solver" : "cg","preconditioner": "sor","relaxation_parameter":0.9,'relative_tolerance':1E-9,'absolute_tolerance':1E-9,"convergence_criterion" : "residual"}})
		

	return l
	
	
