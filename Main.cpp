
#include "ArffImporter.h"
#include "RandomForest.h"
#include <iostream>
#include <omp.h>
#include <time.h>

using namespace std;

int main(int argc, char **argv)
{
	MPI_Comm comm2d;
	int up, down;
	int dims[2];
	int period[2];
	int rank, size;
	time_t start, end;

	if (argc < 2)
	{
		cout << "Run with training file and test file as parameters" << endl;
		return -1;
	}

	period[0] = 1;
	period[1] = 1;
	dims[0] = 0;
	dims[1] = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Dims_create(size, 2, dims);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 1, &comm2d);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Cart_shift(comm2d, 1, 1, &down, &up);

	ArffImporter *trainSetImporter = new ArffImporter(argv[1]);
	ArffImporter *testSetImporter = new ArffImporter(argv[2]);
	RandomForest *classifier = new RandomForest(rank, size);

	time(&start);
	classifier->Train(trainSetImporter->GetInstances(),
					  trainSetImporter->GetFeatures(),
					  trainSetImporter->GetClassAttr(),
					  trainSetImporter->GetNumInstances());
	classifier->Classify(testSetImporter->GetInstances(),
						 testSetImporter->GetNumInstances());

	time(&end);

	if (rank == 0)
		cout << "Time taken is: " << difftime(end, start)
			<< "sec" << endl;

	free(classifier);
	free(trainSetImporter);
	free(testSetImporter);

	MPI_Finalize();
	return 0;
}
