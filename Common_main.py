from mpi4py import MPI
import os
import sys
import getopt

######################################################################################################################
# 						Help command output						     #
######################################################################################################################

def helpcommand():
	print("The following sintax needs to be provided in input for the problem:")
	print("\nData:\n")
	print(" -f (--solveproblem) \t<inputrfile> \tCommand to solver the problem with the data of this .prm file")
	print(" -g (--generateparameterfile) \t<outputfile> \t Command to generate the .prm file for the problem")
	print(" -c (--convergencetest) \t<convergenceiterations> \t Number of convergence iterations to perform")


######################################################################################################################
# 					Control of the input arguments and execution				     #
######################################################################################################################

def main(argv, dir, dir_par):

	# Import of the Parameter file and of the solver connected
	sys.path.append(dir + dir_par)
	sys.path.append(dir)

	import Parameters as ModelPRM
	import Problem_Solver as solver

	try:
		opts, args = getopt.getopt(argv, "hg:f:c:",["help","generateparameterfile=","solveproblem=","convergencetest="])
	except getopt.GetoptError:
		print("Error in calling the script")
		helpcommand()
		sys.exit(2)

	conv = 0

	for opt,arg in opts:

		# Help command execution
		if opt in ("-h", "--help"):
			helpcommand()
			sys.exit(1)

		# Convergence test mode
		elif opt in ("-c", "--convergencetest"):
			conv = int(arg)

	for opt,arg in opts:

		# Generation of the parameter file
		if opt in ("-g", "--generateparameterfile"):
			filename = arg

			if conv > 0:
				ModelPRM.CreateNSprm(filename, True)

			else:
				ModelPRM.CreateNSprm(filename)

			sys.exit(1)

		# Resolution of the problem
		elif opt in ("-f", "--solveproblem"):
			filename = arg

			if conv > 0:
				solver.problemconvergence(filename, conv)

			else:
				solver.problemsolver(filename)

	return 1
