from pyrameters import PRM

#################################################################################
# Tools to control the correct indentation that comments need in parameter file #
#################################################################################

def findinitialspaces(line):

	comm = None
	for ii in line:
		if ii == " ":
			if comm == None:
				comm = " "
			else:
				comm = comm + " "
		else:
			return comm

	return comm



#########################################################################
# Generator of a .prm file output starting from a pyrameters dictionary #
#########################################################################

def generateprmfile(prm, filename):

	f = open(filename, "w")
	print(prm, file=f)
	f.close()

	return 1



####################################################
#        Reader of a .prm file line-by-line        #
####################################################

def readlinesprmfile(filename):

	f1 = open(filename, "r")
	Lines = f1.readlines()
	f1.close()

	Lines.insert(0,"\n")
	Lines.insert(0,"# Listing of Parameters\n")

	return Lines



######################################################################################
# Function that given a comment and an associated line, add the comment to the lines #
######################################################################################

def addcomment(comm, line, commentedlines):

	if not (comm == None):
		if not (comm.find("#") == -1):
			comm = comm + "\n"
			commentedlines.append(comm)

	commentedlines.append(line)

	if not (comm == None):
		if comm.endswith("\n"):
			commentedlines.append("\n")

	return commentedlines



######################################################
#       Writer of a .prm file given the lines        #
######################################################

def writelinesprmfile(commentedlines, filename):

	f = open(filename, "w")
	f.writelines(commentedlines)
	f.close()

	return 1



#############################################################################################
# Reader of parameter file given a filename. The function returns the pyrameters dictionary #
#############################################################################################

def readprmfile(filename):

	if filename.endswith(".prm"):
		with open(filename, 'r') as fileprm:
			prm = PRM(fileprm.read())

	return prm
