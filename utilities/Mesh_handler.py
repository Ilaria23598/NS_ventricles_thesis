import mshr
import dolfin
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import sys

############################################################################################################
# Control the input parameters about the type of mesh to be used and call the generation/import procedures #
############################################################################################################

def MeshHandler(param, it):

	# If it exists a file for the mesh choose the importing procedure
	if param['Domain Definition']['Type of Mesh'] == "File":

		mesh = MeshImportFromFile(param)

	# If a built-in generation is wanted we select the built-in procedure
	elif param['Domain Definition']['Type of Mesh'] == "Built-in":

		mesh = BuiltInMeshGenerator(param, it)

	# No other choices are available, so print an ERROR MESSAGE
	else:
		print("Type of Mesh choice not valid: please choose between File or Built-in options!")
		sys.exit(0)

	# Mesh Generated Return
	return mesh



#######################################################################################################
#			 	      Built-in mesh generation procedure         		      #
#######################################################################################################

def BuiltInMeshGenerator(param, it):

	# Cubic Case Mesh Generation
	if param['Domain Definition']['Built-in Mesh']['Geometry Type'] == "Holed-Cube":

		# Geometrical Information about the Cubes
		L1 = param['Domain Definition']['Built-in Mesh']['Cubic Mesh']['External Edge Length']
		L2 = param['Domain Definition']['Built-in Mesh']['Cubic Mesh']['Internal Edge Length']

		# Domain Generation
		domain = GenerateCubicDomain(L1, L2)

	# Cubic Case Mesh Generation
	if param['Domain Definition']['Built-in Mesh']['Geometry Type'] == "Cube":

		ref = int(param['Domain Definition']['Built-in Mesh']['Mesh Refinement'])*(2**it)
		mesh = UnitCubeMesh(ref,ref,ref)

		return mesh
		
	elif param['Domain Definition']['Built-in Mesh']['Geometry Type'] == "Cube1":

		ref = int(param['Domain Definition']['Built-in Mesh']['Mesh Refinement'])*(2**it)
		mesh = BoxMesh(Point(-1,-1,-1),Point(1,1,1),ref,ref,ref)

		return mesh

	# Cubic Case Mesh Generation
	elif param['Domain Definition']['Built-in Mesh']['Geometry Type'] == "Sphere":

		# Geometrical Information about the Cubes
		r1 = param['Domain Definition']['Built-in Mesh']['Spherical Mesh']['External Radius']
		r2 = param['Domain Definition']['Built-in Mesh']['Spherical Mesh']['Internal Radius']

		# Domain Generation
		domain = GenerateSphericalDomain(r1, r2)

	elif param['Domain Definition']['Built-in Mesh']['Geometry Type'] == "Square":

		# Mesh refinement definition
		ref = int(param['Domain Definition']['Built-in Mesh']['Mesh Refinement'])*(2**it)
		mesh = UnitSquareMesh(ref,ref)
		
		if (MPI.comm_world.Get_rank() == 0):
			print("Mesh Generation Procedure Ended!")

		return mesh
		
	elif param['Domain Definition']['Built-in Mesh']['Geometry Type'] == "Square1":
	        ref = int(param['Domain Definition']['Built-in Mesh']['Mesh Refinement'])*(2**it)
	        mesh = RectangleMesh(Point(-1, -1), Point(1, 1), ref, ref)
	        if (MPI.comm_world.Get_rank() == 0):
	               print("Mesh Generation Procedure Ended!")
			
	        return mesh
  

	else:

		# Error in geometry definition
		print("Mesh Geometrical Type not valid! Control the Built-in meshes available types in the parameter file!")
		sys.exit(0)

	print("Domain Generation Procedure Ended!")

	# Extraction of the refinement number from the parameter file
	nRef = param['Domain Definition']['Built-in Mesh']['Mesh Refinement']

	# Mesh Generation from a generic domain
	mesh = mshr.generate_mesh(domain, nRef)

	if (MPI.comm_world.Get_rank() == 0):
		print("Mesh Generation Procedure Ended!")

	return mesh



#########################################################################################
#		Cubic domain generation procedure given the edge lengths		#
#########################################################################################

def GenerateCubicDomain(L1, L2):

	if L2 == False:
		domain = mshr.Box(dolfin.Point(0,0,0),dolfin.Point(L1,L1,L1))

	else:
		# External box generation
		BoxExt = mshr.Box(dolfin.Point(-L1/2,-L1/2,-L1/2),dolfin.Point(L1/2, L1/2, L1/2))

		# Internal box generation
		BoxInt = mshr.Box(dolfin.Point(-L2/2,-L2/2,-L2/2),dolfin.Point(L2/2, L2/2, L2/2))

		# Domain Generation
		domain = BoxExt - BoxInt

	return domain



#########################################################################################
#		Spherical domain generation procedure given the radius lengths		#
#########################################################################################

def GenerateSphericalDomain(r1, r2):

	# External sphere generation
	SphExt = mshr.Sphere(dolfin.Point(0,0,0), r1)

	# Internal sphere generation
	SphInt = mshr.Sphere(dolfin.Point(0,0,0), r2)

	# Domain Generation
	domain = SphExt - SphInt

	return domain



#######################################################################################################
#				Import mesh from existing file procedure         		      #
#######################################################################################################

def MeshImportFromFile(param):

	# File name recover from
	filename = param['Domain Definition']['Mesh from File']['File Name']

	if not (filename[len(filename)-3:len(filename)] == ".h5"):

		# Error if the extension of the file is not possible
		print("ERROR! The extension of the file cannot be used!")
		sys.exit(0)

	# Mesh initialization
	mesh = Mesh()

	# Mesh .h5 reading
	file = HDF5File(MPI.comm_world, filename, "r")
	file.read(mesh, "/mesh", False)

	file.close()

	if (MPI.comm_world.Get_rank() == 0):
		print("Mesh read from file!")

	return mesh
