import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from pyrameters import PRM

cwd = os.getcwd()
sys.path.append(cwd + '/../utilities/')

import ParameterFile_handler as pfh



#################################################################################
# 	   Generator of parameter file for the Fisher Kolmogorov problem 		#
#################################################################################

def CreateNSprm(filename, conv = False):

	# Generate default parameters
	prm = parametersNS(conv)

	if pfh.generateprmfile(prm,filename) == 1:

		Lines = pfh.readlinesprmfile(filename)
		commentedlines = commentingprmNS(Lines)

		if pfh.writelinesprmfile(commentedlines, filename) == 1:

			print("Parameter file with default values generated")

		else:
			print("Error in parameter file generation!")

	else:

		print("Error in parameter file generation!")



#################################################################################
# 	Definition of the default parameters for the Fisher Kolmogorov problem 	#
#################################################################################

def parametersNS(conv):

	# Create the prm file object
	prm = PRM()

	# PARAMETERS DEFINITION

	# SUBSECTION OF SPATIAL DISCRETIZATION

	prm.add_subsection('Spatial Discretization')

	prm['Spatial Discretization']['Method'] = 'DG-FEM'
	prm['Spatial Discretization']['Polynomial Degree velocity'] = 1
	prm['Spatial Discretization']['Polynomial Degree pressure'] = 1

	# SUBSECTION OF MESH INFORMATION

	prm.add_subsection('Domain Definition')

	prm['Domain Definition']['Type of Mesh'] = 'Built-in'
	prm['Domain Definition']['ID for Ventricles'] = 2
	prm['Domain Definition']['ID for Skull'] = 1
	prm['Domain Definition']['Boundary ID Function Name'] = "boundaries"
	prm['Domain Definition']['Subdomain ID Function Name'] = "subdomains"

	# Definition of a built-in mesh
	prm['Domain Definition'].add_subsection('Built-in Mesh')

	prm['Domain Definition']['Built-in Mesh']['Geometry Type'] = 'Cube'
	prm['Domain Definition']['Built-in Mesh']['Mesh Refinement'] = 20

	# Definition of a cubic mesh
	prm['Domain Definition']['Built-in Mesh'].add_subsection('Cubic Mesh')

	prm['Domain Definition']['Built-in Mesh']['Cubic Mesh']['External Edge Length'] = 0.1
	prm['Domain Definition']['Built-in Mesh']['Cubic Mesh']['Internal Edge Length'] = 0.01


	# Definition of a cubic mesh
	prm['Domain Definition']['Built-in Mesh'].add_subsection('Spherical Mesh')

	prm['Domain Definition']['Built-in Mesh']['Spherical Mesh']['External Radius'] = 0.1
	prm['Domain Definition']['Built-in Mesh']['Spherical Mesh']['Internal Radius'] = 0.01

	# Definition of file information
	prm['Domain Definition'].add_subsection('Mesh from File')

	prm['Domain Definition']['Mesh from File']['File Name'] = "..."


	# SUBSECTION OF TEMPORAL DISCRETIZATION

	prm.add_subsection('Temporal Discretization')

	# Definition of final time and timestep
	prm['Temporal Discretization']['Final Time'] = 1.0
	prm['Temporal Discretization']['Time Step'] = 0.01
	prm['Temporal Discretization']['Problem Periodicity'] = 0.0
	prm['Temporal Discretization']['Theta-Method Parameter'] = 1.0

	# SUBSECTION PARAMETERS OF THE MODEL

	prm.add_subsection('Model Parameters')

	prm['Model Parameters']['mu']=0.1
	prm['Model Parameters']['gammav']=1.0
	prm['Model Parameters']['gammav']=1.0

	prm['Model Parameters']['Isotropic Diffusion'] = 'Yes'

	
	
	

	prm['Model Parameters']['Forcing Terms_1x'] = '0*x[0]'
	prm['Model Parameters']['Forcing Terms_1y'] = '0*x[0]'
	prm['Model Parameters']['Forcing Terms_1z'] = '0*x[0]'
	prm['Model Parameters']['Forcing Terms_2x'] = '0*x[0]'
	prm['Model Parameters']['Forcing Terms_2y'] = '0*x[0]'
	prm['Model Parameters']['Forcing Terms_3x'] = '0*x[0]'
	prm['Model Parameters']['Forcing Terms_3y'] = '0*x[0]'
	prm['Model Parameters']['g1 Termsx']='0*x[0]'
	prm['Model Parameters']['g1 Termsy']='0*x[0]'
	prm['Model Parameters']['g1 Termsz']= '0*x[0]'
	prm['Model Parameters']['g2 Termsx']= '0*x[0]'
	prm['Model Parameters']['g2 Termsy']= '0*x[0]'
	prm['Model Parameters']['g2 Termsz']= '0*x[0]'


    	# SUBSECTION OF BOUNDARY CONDITIONS

	prm.add_subsection('Boundary Conditions')

	

	prm['Boundary Conditions']['Ventricles BCs'] = "Neumann"
	prm['Boundary Conditions']['Skull BCs'] = "Neumann"
	prm['Boundary Conditions']['Input for Ventricles BCs'] = "Constant"
	prm['Boundary Conditions']['Input for Skull BCs'] = "Constant"
	prm['Boundary Conditions']['File Column Name Ventricles BCs '] = "u_vent"	
	prm['Boundary Conditions']['File Column Name Skull BCs '] = "u_skull"
	
	prm['Boundary Conditions']['Skull Dirichlet BCs Value'] = 0
	
	
	prm['Boundary Conditions']['File Column Name Skull BCs (x-component)'] = "ux_skull"

	
	
	# SUBSECTION OF INITIAL CONDITION
	
	prm['Model Parameters']['Initial Condition (Pressure)'] = 0.0
	prm['Model Parameters']['Initial Condition (Velocity)'] = [0,0,0]
	prm['Model Parameters']['Initial Condition from File (Pressure)'] = "No"
	prm['Model Parameters']['Initial Condition from File (Velocity)'] = "No"
	prm['Model Parameters']['Initial Condition File Name'] = "..."
	prm['Model Parameters']['Name of IC Function in File'] = "u0"
	

	# SUBSECTION OF LINEAR SOLVER

	prm.add_subsection("Linear Solver")

	prm['Linear Solver']['Type of Solver'] = 'Default'
	prm['Linear Solver']['Iterative Solver'] = "..."
	prm['Linear Solver']['Preconditioner'] = "..."
	prm['Linear Solver']['User-Defined Preconditioner'] = 'No'

	# SUBSECTION OF OUTPUT FILE

	prm.add_subsection('Output')

	prm['Output']['Output XDMF File Name'] = "..."

	# SUBSECTION OF CONVERGENCE TEST

	if (conv):
		prm.add_subsection('Convergence Test')

		prm['Convergence Test']['Exact Solution Velocity x'] = '0*x[0]'
		prm['Convergence Test']['Exact Solution Velocity y'] = '0*x[0]'
		prm['Convergence Test']['Exact Solution Pressure'] = '0*x[0]'


	return prm



#################################################################################################################
# 	   		Generator of comments for the Fisher Kolmogorov parameters file		 		#
#################################################################################################################

def commentingprmNS(Lines):

	commentedlines = []

	for line in Lines:

		comm = pfh.findinitialspaces(line)

		if not (line.find("set Method") == -1):
			comm = comm + "# Decide the type of spatial discretization method to apply (DG-FEM/CG-FEM)"

		if not (line.find("set Method") == -1):
			comm = comm + "# Decide the polynomial degree of the FEM approximation"

		if not (line.find("set Type of Mesh") == -1):
			comm = comm + "# Decide the type of mesh to use in your simulation: File/Built-in"

		if not (line.find("set ID for Ventricles") == -1):
			comm = comm + "# Set the value of boundary ID of ventricles"

		if not (line.find("set ID for Skull") == -1):
			comm = comm + "# Set the value of boundary ID of skull"

		if not (line.find("set Boundary ID Function Name") == -1):
			comm = comm + "# Set the name of the function containing the boundary ID"

		if not (line.find("set Subdomain ID Function Name") == -1):
			comm = comm + "# Set the name of the function containing the subdomain ID"

		if not (line.find("set Geometry Type") == -1):
			comm = comm + "# Decide the type of geometrical built-in object: Cube/Sphere/Square/Square1"

		if not (line.find("set Mesh Refinement") == -1):
			comm = comm + "# Refinement value of the mesh"

		if not (line.find("set External Edge Length") == -1):
			comm = comm + "# Length of the external cube edge [m]"

		if not (line.find("set Internal Edge Length") == -1):
			comm = comm + "# Length of the internal cube edge [m]"

		if not (line.find("set External Radius") == -1):
			comm = comm + "# Length of the external sphere radius [m]"

		if not (line.find("set Internal Radius") == -1):
			comm = comm + "# Length of the internal sphere radius [m]"

		if not (line.find("set File Name") == -1):
			comm = comm + "# Name of the file containing the mesh. Possible extensions: .h5"

		if not (line.find("set Final Time") == -1):
			comm = comm + "# Final time of the simulation [years]"

		if not (line.find("set Time Step") == -1):
			comm = comm + "# Time step of the problem [years]"

		if not (line.find("set Problem Periodicity") == -1):
			comm = comm + "# Periodicity of the BCs [years]"

		if not (line.find("set mu") == -1):
			comm = comm + "# Diffusion Constant of the Fluid Network "

		

		if not (line.find("set Time Derivative Coefficient") == -1):
			comm = comm + "# Time Derivative Coefficient of the Fluid Network "

		if not (line.find("set Isotropic Diffusion") == -1):
			comm = comm + "# Isotropic Diffusion Tensors assumption: Yes/No"

		if not (line.find("set") == -1) and not(line.find("Coupling Parameter") == -1):
			comm = comm + "# Coupling Parameters between the Fluid Networks "


		if not(line.find("set Ventricles BCs") == -1):
			comm = comm + "# Type of Boundary Condition imposed on the Ventricular Surface: Dirichlet/Neumann"

		if not(line.find("set Skull BCs") == -1):
			comm = comm + "# Type of Boundary Condition imposed on the Skull Surface: Dirichlet/Neumann"

		if not(line.find("set Initial Condition from File") == -1):
			comm = comm + "# Enable the reading of an initial condition from file"

		if not(line.find("set Initial Condition File Name") == -1):
			comm = comm + "# Name of the file containing the initial condition"

		if not(line.find("set Name of IC Function in File") == -1):
			comm = comm + "# Name of the function containing the initial condition in the file"

		if not(line.find("set Input for Ventricles BCs") == -1):
			comm = comm + "# Type of Input for the imposition of Boundary Condition on the Ventricular Surface: Constant/File/Expression"

		if not(line.find("set Input for Skull BCs") == -1):
			comm = comm + "# Type of Input for the imposition of Boundary Condition imposed on the Skull Surface: Constant/File/Expression"

		if not(line.find("set File Column Name") == -1):
			comm = comm + "# Set the Column Name where is stored the BCs in the .csv file (associated to a column time)"

		if not(line.find("set") == -1) and not(line.find("Dirichlet BCs Value") == -1):
			comm = comm + "# Boundary Condition value to be imposed [m]"

		if not(line.find("set") == -1) and not(line.find("Dirichlet BCs Value (Pressure)") == -1):
			comm = comm + "# Boundary Condition value to be imposed [Pa]: insert the constant value or the file name"

		if not(line.find("set") == -1) and not(line.find("Neumann BCs Value (Stress") == -1):
			comm = comm + "# Boundary Condition value to be imposed [Pa]"

		if not(line.find("set") == -1) and not(line.find("Neumann BCs Value (Flux)") == -1):
			comm = comm + "# Boundary Condition value to be imposed [m^3/s]: insert the constant value or the file name"

		if not(line.find("set Output XDMF File Name") == -1):
			comm = comm + "# Output file name (The relative/absolute path must be indicated!)"

		if not(line.find("set Initial Condition (Pressure)") == -1):
			comm = comm + "# Initial condition of a pressure value [Pa]"

		if not(line.find("set Initial Condition") == -1):
			comm = comm + "# Initial condition "

		if not(line.find("set") == -1) and not(line.find("Exact Solution") == -1):
			comm = comm + "# Exact solution of the test problem"
	

		if not(line.find("set Type of Solver") == -1):
			comm = comm + "# Choice of linear solver type: Default/Iterative Solver"

		if not(line.find("set Iterative Solver") == -1):
			comm = comm + "# Choice of iterative solver type. The available options are: \n"
			comm = comm + "  " + "#   gmres - cg - minres - tfqmr - richardson - bicgstab - nash - stcg"

		if not(line.find("set Preconditioner") == -1):
			comm = comm + "# Choice of preconditioner type. The available options are: \n"
			comm = comm + "  " + "#   ilu - icc - jacobi - bjacobi - sor - additive_schwarz - petsc_amg - hypre_amg - \n"
			comm = comm + "  " + "#   hypre_euclid - hypre_parasails - amg - ml_amg - none"

		if not(line.find("set User-Defined Preconditioner") == -1):
			comm = comm + "# Choice of using the user defined block preconditioner: Yes/No"

		if not(line.find("set Theta-Method Parameter") == -1):
			comm = comm + "# Choice of the value of the parameter theta: IE(1) - CN(0.5) - EE(0)"

		commentedlines = pfh.addcomment(comm, line, commentedlines)

	return commentedlines
