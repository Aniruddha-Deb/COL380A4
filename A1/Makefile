CC=g++
CFLAGS=-I. -fopenmp -O3 -std=c++11
LDFLAGS=-L.

TESTMAT=test/input

apr1:
	$(CC) $(CFLAGS) approach_1.cpp library.so $(LDFLAGS) -o approach_1

apr2:
	$(CC) $(CFLAGS) approach_2.cpp library.so $(LDFLAGS) -o approach_2

apr3:
	$(CC) $(CFLAGS) approach_3.cpp library.so $(LDFLAGS) -o approach_3

apr3_test:
	./approach_3 $(TESTMAT) apr3_output/output
	./mattxt apr3_output/output 2 > apr3_output/output.txt
	diff data/output.txt apr3_output/output.txt

apr2_test:
	./approach_2 $(TESTMAT) apr2_output/output
	./mattxt apr2_output/output 2 > apr2_output/output.txt
	diff data/output.txt apr2_output/output.txt

apr1_test:
	./approach_1 $(TESTMAT) apr1_output/output
	./mattxt apr1_output/output 2 > apr1_output/output.txt
	diff apr0_output/output.txt apr1_output/output.txt

apr0:
	$(CC) $(CFLAGS) approach_0.cpp $(LDFLAGS) -o approach_0
	./approach_0 test/input apr0_output/output
	./mattxt apr0_output/output 2 > apr0_output/output.txt
	diff test/output apr0_output/output
