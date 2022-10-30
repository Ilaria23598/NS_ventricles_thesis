import mshr
import meshio
import math
import dolfin
import mpi4py
import pandas as pd
from fenics import *
import numpy as np
import sys
import getopt
import time


#######################################################################################################################################
#                                                       Conversion from .vtu to .xml mesh                                             #
#######################################################################################################################################

def vtutoxmlmesh_homog_skull_vent(param):

	# Read vtu initial mesh
	meshxml = meshio.read(param["InputFileName"])

	# Creation of XML file name
	XMLMeshFile = param["InputFileName"][0:len(param["InputFileName"])-4] + ".xml"
	XMLMeshFileBDD = param["InputFileName"][0:len(param["InputFileName"])-4] + "_CellBDI.xml"

	# Creation of XML Mesh
	meshio.write_points_cells(XMLMeshFile, points = meshxml.points, cells = meshxml.cells, cell_data = {"CellBDI": meshxml.cell_data["CellBDI"]})
	print("\n")
	print("XML File Mesh Created: ", XMLMeshFile)
	print("XML File for Boundary Tags Created: ", XMLMeshFileBDD)
	print("\n")

	return XMLMeshFile, XMLMeshFileBDD



#######################################################################################################################################
#	                                               Conversion from .xml to .h5 mesh                                               #
#######################################################################################################################################

def xmltoh5_homog_skull_vent(XMLMeshFile, XMLMeshFileBDD, param):

	# Mesh Importation
	mesh = Mesh(XMLMeshFile)

	# Importing the identifier as cell element
	cellbdd = MeshFunction("size_t", mesh, XMLMeshFileBDD)

	# Computing the mesh dimension
	D = mesh.topology().dim()

	# Transferring the information over the faces
	facebdd = MeshFunction("size_t", mesh, D-1)

	bound = np.zeros(len(facebdd))
	for cell in cells(mesh):
		for ii in facets(cell):
			bound[ii.index()] = bound[ii.index()] + 1
			if bound[ii.index()] > 1:
				facebdd[ii.index()] = 0
			else:
				facebdd[ii.index()] = cellbdd[cell.index()]

	print("Facets Indicator Created")

	# Transferring the information over the vertices
	hdf = HDF5File(MPI.comm_world, param["OutputFileName"], "w")
	hdf.write(mesh, "/mesh")
	hdf.write(cellbdd, "/CellBDI")
	hdf.write(facebdd, "/boundaries")

	print("Mesh in h5 format created!")



#######################################################################################################################################
#	                                               Conversion from .xml to .h5 mesh                                               #
#######################################################################################################################################

def xdmftoh5_heterog(XDMFFileMesh, XDMFFileSubdomains, XDMFFileBoundaries, OutputFileName):

	# Read the mesh from XDMF
	mesh = dolfin.Mesh()

	infile = dolfin.XDMFFile(XDMFFileMesh)
	infile.read(mesh)

	# Compute the topological dimension of the mesh
	n = mesh.topology().dim()

	# Read the subdomains from the xdmf file
	subdomains = dolfin.MeshFunction("size_t", mesh, n)

	infile = dolfin.XDMFFile(XDMFFileSubdomains)
	infile.read(subdomains, "subdomains")

	# Read the boundaries from the xdmf file of triangles
	mesh_triangles = dolfin.Mesh()

	infile = dolfin.XDMFFile(XDMFFileBoundaries)
	infile.read(mesh_triangles)

	bdd = dolfin.MeshFunction("size_t", mesh_triangles, n-1)
	infile.read(bdd, "boundaries")

	# Construct the midpoints table of the boundary's triangles
	mid = [cell.midpoint()[:] for cell in cells(mesh_triangles)]
	middf = pd.DataFrame(mid, columns=['0','1','2'])

	# Construct the triangular elements for the mesh
	boundaries = dolfin.MeshFunction("size_t", mesh, n-1, 0)

	print("Projection of the boundary IDs started!")

	t = time.time()
	for face in dolfin.facets(mesh):
		#app = np.where((mid[1:3]==face.midpoint()[:]).all(axis=1))
		app = middf.loc[(middf['0']==face.midpoint()[0]) & (middf['1']==face.midpoint()[1]) & (middf['2']==face.midpoint()[2])]


		# If we found the triangle in the boundaries we change the ID
		if app.shape[0] == 1:
			boundaries.array()[face.index()] = bdd.array()[app.index[0]]
			middf.drop(app.index[0], inplace=True)

		if face.index()%10000 == 0:
			advancement = ' - Status (last 10000 computed in ' + str(math.trunc((time.time()-t)*10000)/10000) + ' sec) : ' + str(math.trunc(face.index()/boundaries.size()*10000)/100) + ' % (Remained: ' + str(middf.shape[0]) + ')'
			t = time.time()
			print(advancement)

	print("Projection of the boundary IDs finished!")

	# Save the .h5 file
	hdf = dolfin.HDF5File(mesh.mpi_comm(), OutputFileName, "w")
	hdf.write(mesh, "/mesh")
	hdf.write(subdomains, "/subdomains")
	hdf.write(boundaries, "/boundaries")
	hdf.close()

	print("Mesh .h5 created")

