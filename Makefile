SHELL = /bin/sh
# Enable debug options
# CFLAGS = -g -Wall -fopenmp
# Enable best optimization options
CFLAGS = -Ofast -march=native -mtune=native -fopenmp -std=c++11
CC = mpic++
OBJECTS = Unity.o ArffImporter.o TreeFactory.o RandomForest.o

exec: ${OBJECTS} Main.cpp
	$(CC) ${CFLAGS} -o $@ ${OBJECTS} Main.cpp

Unity.o: Unity.cpp Unity.h
	$(CC) ${CFLAGS} -c Unity.cpp

ArffImporter.o: ArffImporter.cpp ArffImporter.h Unity.h
	$(CC) ${CFLAGS} -c ArffImporter.cpp

TreeFactory.o: TreeFactory.cpp TreeFactory.h Unity.h
	$(CC) ${CFLAGS} -c TreeFactory.cpp

RandomForest.o: RandomForest.cpp RandomForest.h TreeFactory.h
	$(CC) ${CFLAGS} -c RandomForest.cpp

clean:
	-rm -f *.o *.h.gch exec
