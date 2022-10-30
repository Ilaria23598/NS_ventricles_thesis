import mshr
import dolfin
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import sys
import pandas as pd

############################################################################################################
# 		Saving an HDF5 file with the solution of the MPT problem				   #
############################################################################################################

def SaveDarcySolution(outputfilename, mesh, x):

	# Output file for tensorial diffusion
	file = HDF5File(MPI.comm_world, outputfilename, "w")

	pc, pa, pv = x.split(deepcopy=True)

	# Writing mesh and functions on the outputfile
	file.write(mesh, "/mesh")
	file.write(pc, "/pc0")
	file.write(pa, "/pa0")
	file.write(pv, "/pv0")

	file.close()

def ImportICfromFile(filename, mesh, x, x_name):

	# File name recover from parameters
	file = HDF5File(MPI.comm_world, filename, "r")

	# Read the function from the file
	x_name_imp = "/" + x_name
	file.read(x, x_name_imp)

	file.close()

	if (MPI.comm_world.Get_rank() == 0):
		mess = "Initial Condition " + x_name + "imported from File!"
		print(mess)

	return x
