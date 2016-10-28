import matplotlib.pyplot as plt
import sys

NUM_BINS = 500
ID = 0


def load_histogram():
	matrix = []
	with open("data/std_train.pickle_histo.csv") as file:
		for line in file.readlines():
			matrix.append(map(int, line.split(",")))
	return matrix

if __name__ == "__main__":
	
	#ID = int(sys.argv[1])
	matrix = load_histogram()

	#Young people
	Y = matrix[0]
	X = range(len(Y))
	plt.plot(X, Y, color="blue")
	Y = matrix[4]
	X = range(len(Y))
	plt.plot(X, Y, color="blue")
	Y = matrix[1]
	X = range(len(Y))
	plt.plot(X, Y, color="blue")
	Y = matrix[12]
	X = range(len(Y))
	plt.plot(X, Y, color="blue")

	#Midage
	Y = matrix[9]
	X = range(len(Y))
	plt.plot(X, Y, color="green")
	Y = matrix[10]
	X = range(len(Y))
	plt.plot(X, Y, color="green")
	Y = matrix[80]
	X = range(len(Y))
	plt.plot(X, Y, color="green")
	Y = matrix[120]
	X = range(len(Y))
	plt.plot(X, Y, color="green")

	#Old people
	Y = matrix[3]
	X = range(len(Y))
	plt.plot(X, Y, color="red")
	Y = matrix[8]
	X = range(len(Y))
	plt.plot(X, Y, color="red")
	Y = matrix[27]
	X = range(len(Y))
	plt.plot(X, Y, color="red")
	Y = matrix[51]
	X = range(len(Y))
	plt.plot(X, Y, color="red")

	plt.show()