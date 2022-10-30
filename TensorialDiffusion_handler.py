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

def SavePermeabilityTensorinHomMeshes(outputfilename, mesh, boundaries, K):

	# Output file for tensorial diffusion
	file = HDF5File(MPI.comm_world, outputfilename, "w")

	# Writing mesh and functions on the outputfile 
	file.write(mesh, "/mesh")
	file.write(boundaries, "/boundaries")
	file.write(K, "/K")

	file.close()

def ImportPermeabilityTensor(param, mesh, K):

	# File name recover from parameters
	filename = param['Domain Definition']['Mesh from File']['File Name']

	file = HDF5File(MPI.comm_world, filename, "r")

	# Read the function from the file
	file.read(K, "/K")

	file.close()
	
	print("Permeability Tensor Imported!")
	return K
