
#include "ArffImporter.h"
#include "RandomForest.h"
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	int rank, size;
	if (argc < 2) {
		cout << "Please run with training file and test file as parameters" << endl;
		return -1;
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ArffImporter* trainSetImporter = new ArffImporter(argv[1]);
	ArffImporter* testSetImporter = new ArffImporter(argv[2]);

	RandomForest* classifier = new RandomForest(rank, size);
	classifier->Train(trainSetImporter->GetInstances(),
					 trainSetImporter->GetFeatures(),
					 trainSetImporter->GetClassAttr(),
					 trainSetImporter->GetNumInstances());
	classifier->Classify(testSetImporter->GetInstances(),
						testSetImporter->GetNumInstances());

	free(classifier);
	free(trainSetImporter);
	free(testSetImporter);

	MPI_Finalize();
	return 0;
}
