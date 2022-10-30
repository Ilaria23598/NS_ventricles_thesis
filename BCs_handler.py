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

def ImportFaceFunction(param, mesh):

	if param["Domain Definition"]["Type of Mesh"] == "File":

		BoundaryID = ImportFaceFunctionFromFile(param, mesh)

	elif param["Domain Definition"]["Type of Mesh"] == "Built-in":

		if param["Domain Definition"]["Built-in Mesh"]["Geometry Type"] == "Square":

			BoundaryID = FaceFunctionForSquareCavity(param, mesh)
		if param["Domain Definition"]["Built-in Mesh"]["Geometry Type"] == "Square1":

			BoundaryID = FaceFunctionForSquareCavity(param, mesh)

		if param["Domain Definition"]["Built-in Mesh"]["Geometry Type"] == "Cube":

			BoundaryID = FaceFunctionForCubeCavity(param, mesh)
		if param["Domain Definition"]["Built-in Mesh"]["Geometry Type"] == "Cube1":

			BoundaryID = FaceFunctionForCubeCavity(param, mesh)

	return BoundaryID



############################################################################################################
# Control the input parameters about the type of mesh to be used and call the generation/import procedures #
############################################################################################################

def ImportFaceFunctionFromFile(param, mesh):

	# File name recover from parameters
	filename = param['Domain Definition']['Mesh from File']['File Name']

	file = HDF5File(MPI.comm_world, filename, "r")

	# Detect mesh dimension
	D = mesh.topology().dim()

	# Definition the face function
	BoundaryID = MeshFunction("size_t", mesh, D-1)

	# Read the function from the file
	bddname = "/" + param['Domain Definition']['Boundary ID Function Name']
	file.read(BoundaryID, bddname)

	file.close()

	return BoundaryID


##################################################################################################################
# 				Generation of facet IDs for the square cavity problem				 #
##################################################################################################################

def FaceFunctionForSquareCavity(param, mesh):

	# Detect mesh dimension
	D = mesh.topology().dim()

	# Definition the face function
	BoundaryID = MeshFunction("size_t", mesh, D-1)
	BoundaryID.set_all(0)

	for face in facets(mesh):

		# Tag for {y = 1}
		if abs(face.midpoint()[1] - 1) < 1e-4:
			BoundaryID.set_value(face.index(),2)

		# Tag for {y = 0}
		if abs(face.midpoint()[1]) < 1e-4:
			BoundaryID.set_value(face.index(),1)

		# Tag for {x = 1}
		if abs(face.midpoint()[0] - 1) < 1e-4:
			BoundaryID.set_value(face.index(),1)

		# Tag for {x = 0}
		if abs(face.midpoint()[0]) < 1e-4:
			BoundaryID.set_value(face.index(),1)

	return BoundaryID


##################################################################################################################
# 				Generation of facet IDs for the square cavity problem				 #
##################################################################################################################

def FaceFunctionForCubeCavity(param, mesh):

	# Detect mesh dimension
	D = mesh.topology().dim()

	# Definition the face function
	BoundaryID = MeshFunction("size_t", mesh, D-1)
	BoundaryID.set_all(0)

	for face in facets(mesh):

		# Tag for {z = 1}
		if abs(face.midpoint()[2] - 1) < 1e-4:
			BoundaryID.set_value(face.index(),2)

		# Tag for {z = 0}
		if abs(face.midpoint()[2]) < 1e-4:
			BoundaryID.set_value(face.index(),1)

		# Tag for {y = 1}
		if abs(face.midpoint()[1] - 1) < 1e-4:
			BoundaryID.set_value(face.index(),1)

		# Tag for {y = 0}
		if abs(face.midpoint()[1]) < 1e-4:
			BoundaryID.set_value(face.index(),1)

		# Tag for {x = 1}
		if abs(face.midpoint()[0] - 1) < 1e-4:
			BoundaryID.set_value(face.index(),1)

		# Tag for {x = 0}
		if abs(face.midpoint()[0]) < 1e-4:
			BoundaryID.set_value(face.index(),1)

	return BoundaryID



############################################################################################################
# 				Imposition of BCs values for 1D variables				   #
############################################################################################################

def FindBoundaryConditionValue1D(BCsType, BCsValue, BCsColumnName, time, period):

	if (BCsType == "Constant"):

		BCs = Constant(BCsValue)

	elif (BCsType == "File"):

		if (BCsValue[len(BCsValue)-4:len(BCsValue)] == ".csv"):

			df = pd.read_csv(BCsValue)
			BCs = Constant(df[abs(df["time"]-time%period)<1e-8][BCsColumnName].iloc[0])

	elif (BCsType == "Expression"):

		BCs = Expression(BCsValue, degree=6, t=time)

	else:
		print("The choice of BCs type provided is not available!")
		sys.exit(0)

	return BCs



############################################################################################################
# 				Imposition of BCs values for 2D variables				   #
############################################################################################################

def FindBoundaryConditionValue2D(BCsType, BCsValueX, BCsValueY, BCsColumnNameX, BCsColumnNameY, time, period):

	if (BCsType == "Constant"):

		BCs = Constant((BCsValueX, BCsValueY))

	elif (BCsType == "File"):

		if (BCsValueX[len(BCsValueX)-4:len(BCsValueX)] == ".csv") and (BCsValueY[len(BCsValueY)-4:len(BCsValueY)] == ".csv"):

			dfx = pd.read_csv(BCsValueX)
			dfy = pd.read_csv(BCsValueY)

			BCs = Constant((dfx[abs(dfx["time"]-time%period)<1e-8][BCsColumnNameX].iloc[0], dfy[abs(dfy["time"]-time%period)<1e-8][BCsColumnNameY].iloc[0]))

	elif (BCsType == "Expression"):

		BCs = Expression((BCsValueX,BCsValueY), degree=6, t=time)

	else:
		print("The choice of BCs type provided is not available!")
		sys.exit(0)

	return BCs



############################################################################################################
# 				Imposition of BCs values for 3D variables				   #
############################################################################################################

def FindBoundaryConditionValue3D(BCsType, BCsValueX, BCsValueY, BCsValueZ, BCsColumnNameX, BCsColumnNameY, BCsColumnNameZ, time, period):

	if (BCsType == "Constant"):

		BCs = Constant((BCsValueX, BCsValueY, BCsValueZ))

	elif (BCsType == "File"):

		if (BCsValueX[len(BCsValueX)-4:len(BCsValueX)] == ".csv") and (BCsValueY[len(BCsValueY)-4:len(BCsValueY)] == ".csv") and (BCsValueZ[len(BCsValueZ)-4:len(BCsValueZ)] == ".csv"):

			dfx = pd.read_csv(BCsValueX)
			dfy = pd.read_csv(BCsValueY)
			dfz = pd.read_csv(BCsValueZ)

			BCs = Constant((dfx[abs(dfx["time"]-time%period)<1e-8][BCsColumnNameX].iloc[0], dfy[abs(dfy["time"]-time%period)<1e-8][BCsColumnNameY].iloc[0], dfz[abs(dfz["time"]-time%period)<1e-8][BCsColumnNameZ].iloc[0]))

	elif (BCsType == "Expression"):

		BCs = Expression((BCsValueX,BCsValueY,BCsValueZ), degree=6, t=time)

	else:
		print("The choice of BCs type provided is not available!")
		sys.exit(0)

	return BCs

###########################################################################################################
#					Construction of Measures					  #
###########################################################################################################

def MeasuresDefinition(param, mesh, BoundaryID):

	ds_vent = Measure('ds', domain = mesh, subdomain_data = BoundaryID, subdomain_id = int(param['Domain Definition']['ID for Ventricles']))
	ds_skull = Measure('ds', domain = mesh, subdomain_data = BoundaryID, subdomain_id = int(param['Domain Definition']['ID for Skull']))

	return ds_vent, ds_skull
