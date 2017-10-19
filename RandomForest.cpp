
#include "RandomForest.h"

RandomForest::RandomForest()
{
	MPI_Init(nullptr, nullptr);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

RandomForest::~RandomForest()
{
	// Destory trees
	if (root != nullptr)
	{
		for (unsigned int i = 0; i < numTrees; i++)
			treeFactory.DestroyNode(root[i]);
		free(root);
		root = nullptr;
	}

	MPI_Finalize();
}

void RandomForest::Train(const Instance * instanceTable,
						 const vector < NumericAttr > &fv,
						 const vector < char *>&cv,
						 const unsigned int numInstances)
{
	classVec = cv;
	featureVec = fv;

	if (NUM_TREES % size <= 0)
	{
		numTrees = NUM_TREES / size;
	} else {
		numTrees = NUM_TREES / size + 1;
		if (rank == size - 1) {
			numTrees = NUM_TREES - numTrees * (size - 1);
		}
	}

	root = (TreeNode **) malloc(numTrees * sizeof(TreeNode *));
	treeFactory.Init(fv, cv, instanceTable, numInstances);

#pragma omp parallel for schedule(static)
	for (unsigned int treeId = 0; treeId < numTrees; treeId++)
	{
		root[treeId] = treeFactory.Generate(RANDOM_FEATURE_SET_SIZE);
	}
}

void RandomForest::Classify(const Instance * instanceTable,
							const unsigned int numInstances)
{
	unsigned int correctCounter = 0;
	unsigned int *votes = (unsigned int *)
		calloc(classVec.size() * numInstances, sizeof(unsigned int));

#pragma omp parallel for schedule(static)
	for (unsigned int i = 0; i < numInstances; i++)
		Classify(instanceTable[i], votes, i);

	if (rank == 0)
		MPI_Reduce(MPI_IN_PLACE, votes, classVec.size() * numInstances,
							   MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(votes, nullptr, classVec.size() * numInstances,
							   MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		Analysis(votes, instanceTable, correctCounter, numInstances);
	}

	free(votes);
	votes = nullptr;
}

inline void RandomForest::Analysis(unsigned int *votes,
								   const Instance * instanceTable,
								   unsigned int correctCounter,
								   const unsigned int numInstances)
{
#pragma omp parallel for reduction(+: correctCounter) schedule(dynamic)
	for (unsigned int i = 0; i < numInstances; i++)
	{
		unsigned short predictedClassIndex =
			getIndexOfMax(votes + i * classVec.size(), classVec.size());
		if (predictedClassIndex == instanceTable[i].classIndex)
			correctCounter++;
	}

	double correctRate = (double) correctCounter / (double) numInstances;
	double incorrectRate = 1.0 - correctRate;

	printf("Correct rate: %f\n", correctRate);
	printf("Incorrect rate: %f\n", incorrectRate);
}

inline void RandomForest::Classify(const Instance & instance,
								   unsigned int *votes,
								   const unsigned int instId)
{

	for (unsigned int treeId = 0; treeId < numTrees; treeId++)
	{
		TreeNode *node = root[treeId];
		if (node == nullptr)
			continue;

		while (node->left != nullptr || node->right != nullptr)
		{
			TreeNode *tmp;
			if (instance.featureAttrArray[node->featureIndex] >=
				node->threshold)
			{
				tmp = node->right;
			} else
			{
				tmp = node->left;
			}
			if (tmp == nullptr)
				break;
			else
				node = tmp;
		}

		votes[instId * classVec.size() + node->classIndex]++;
	}
}
