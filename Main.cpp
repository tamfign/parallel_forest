
#include "ArffImporter.h"
#include "RandomForest.h"

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(nullptr, nullptr);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ArffImporter trainSetImporter;
	trainSetImporter.Read("Dataset/train/train-first10.arff");

	ArffImporter testSetImporter;
	testSetImporter.Read("Dataset/test/dev-first10.arff");

	RandomForest* classifier = new RandomForest(rank, size);
	classifier->Train(trainSetImporter.GetInstances(),
					 trainSetImporter.GetFeatures(),
					 trainSetImporter.GetClassAttr(),
					 trainSetImporter.GetNumInstances());
	classifier->Classify(testSetImporter.GetInstances(),
						testSetImporter.GetNumInstances());

	free(classifier);

	MPI_Finalize();
	return 0;
}
