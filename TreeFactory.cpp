
#include "TreeFactory.h"

TreeFactory::~TreeFactory()
{

}

TreeFactory::TreeFactory(const vector < NumericAttr > &fv,
					   const vector < char *>&cv,
					   const Instance * it, const unsigned int numInstances)
{
	featureVec = fv;
	classVec = cv;
	instanceTable = it;
	numInstTotal = numInstances;
	numFeaTotal = featureVec.size();
	numClasses = classVec.size();
}

TreeNode *TreeFactory::Generate(const unsigned int numFeaToSelect)
{
	numFeaToTry = numFeaToSelect;

	unsigned int *featureIndexArray =
		(unsigned int *) malloc(numFeaTotal * sizeof(unsigned int));
	for (unsigned int i = 0; i < numFeaTotal; i++)
		featureIndexArray[i] = i;

	MiniInstance *miniInstanceArr =
		(MiniInstance *) malloc(numInstTotal * sizeof(MiniInstance));
	unsigned int *initialClassDist =
		(unsigned int *) calloc(numClasses, sizeof(unsigned int));
	for (unsigned int i = 0; i < numInstTotal; i++)
	{
		// Get overall distribution
		initialClassDist[instanceTable[i].classIndex]++;
		// Init data indices and copy class indices
		miniInstanceArr[i].instanceIndex = i;
		miniInstanceArr[i].classIndex = instanceTable[i].classIndex;
	}

	TreeNode *root = Split(miniInstanceArr,
						   featureIndexArray,
						   initialClassDist,
						   numInstTotal,
						   0);

	free(initialClassDist);
	initialClassDist = nullptr;
	free(miniInstanceArr);
	miniInstanceArr = nullptr;
	free(featureIndexArray);
	featureIndexArray = nullptr;

	return root;
}

TreeNode *TreeFactory::Split(MiniInstance * miniInstanceArr,
							 unsigned int *featureIndexArray,
							 const unsigned int *parentClassDist,
							 const unsigned int numInstances,
							 unsigned int height)
{
	double infoGainMax = 0;

	if (numInstances < MIN_NODE_SIZE)
		return nullptr;
	if (numInstances < MIN_NODE_SIZE_TO_SPLIT)
	{
		TreeNode *leaf = new TreeNode;
		leaf->left = nullptr;
		leaf->right = nullptr;
		leaf->classIndex = getIndexOfMax(parentClassDist, numClasses);

		return leaf;
	}

	double entropyParent = ComputeEntropy(parentClassDist, numInstances);
	if (entropyParent <= 0.0)
	{
		TreeNode *leaf = new TreeNode;
		leaf->left = nullptr;
		leaf->right = nullptr;
		leaf->classIndex = getIndexOfMax(parentClassDist, numClasses);

		return leaf;
	}

	unsigned int selectedFeaIndex;
	double selectedThreshold;

	unsigned int leftSize, rightSize;
	unsigned int selectedLeftSize, selectedRightSize;

	// Init child class distribution vector
	unsigned int *leftDist;
	unsigned int *rightDist;
	leftDist = (unsigned int *) malloc(numClasses * sizeof(unsigned int));
	rightDist = (unsigned int *) malloc(numClasses * sizeof(unsigned int));

	unsigned int *selectedLeft;
	unsigned int *selectedRight;
	selectedLeft = (unsigned int *) malloc(numClasses * sizeof(unsigned int));
	selectedRight = (unsigned int *) malloc(numClasses * sizeof(unsigned int));

	// Store sorted values of that feature with indices
	MiniInstance *selectedMiniInstanceArr =
		(MiniInstance *) malloc(numInstances * sizeof(MiniInstance));

	unsigned int numFeaTried = 0;
	unsigned int numRestFea = numFeaTotal;
	bool gainFound = false;

	// Find the best split feature and threshold
	while ((numFeaTried++ < numFeaToTry || !gainFound) && numRestFea > 0)
	{
		// Sample (note max of rand() is around 32000)
		unsigned int randPos = rand() % numRestFea;
		unsigned int randFeaIndex = featureIndexArray[randPos];

		// Swap
		featureIndexArray[randPos] = featureIndexArray[--numRestFea];
		featureIndexArray[numRestFea] = randFeaIndex;

		// Get all values of that feature with indices and sort them.
		for (unsigned int i = 0; i < numInstances; i++)
			miniInstanceArr[i].featureValue =
				instanceTable[miniInstanceArr[i].instanceIndex].
				featureAttrArray[randFeaIndex];
		sort(miniInstanceArr, miniInstanceArr + numInstances, Compare);

		memset(leftDist, 0, numClasses * sizeof(unsigned int));
		memmove(rightDist, parentClassDist, numClasses * sizeof(unsigned int));

		bool featureIndexStored = false;
		for (unsigned int candidateId = 1;
			 candidateId < numInstances; candidateId++)
		{
			unsigned int preCandidateId = candidateId - 1;

			leftDist[miniInstanceArr[preCandidateId].classIndex]++;
			rightDist[miniInstanceArr[preCandidateId].classIndex]--;

			if (miniInstanceArr[preCandidateId].featureValue <
				miniInstanceArr[candidateId].featureValue)
			{
				double splitThreshold =
					(miniInstanceArr[preCandidateId].featureValue +
					 miniInstanceArr[candidateId].featureValue) / 2.0;
				leftSize = candidateId;
				rightSize = numInstances - candidateId;

				double infoGain = entropyParent;

				// Compute entropy of children
				double entropyChild = ComputeEntropy(leftDist, leftSize);
				infoGain -= leftSize / (double) numInstances *entropyChild;

				entropyChild = ComputeEntropy(rightDist, rightSize);
				infoGain -= rightSize / (double) numInstances *entropyChild;

				if (infoGainMax < infoGain)
				{
					if (!featureIndexStored)
					{
						memmove(selectedMiniInstanceArr,
								miniInstanceArr,
								numInstances * sizeof(MiniInstance));

						selectedFeaIndex = randFeaIndex;
						featureIndexStored = true;
					}

					memmove(selectedLeft, leftDist,
							numClasses * sizeof(unsigned int));
					memmove(selectedRight, rightDist,
							numClasses * sizeof(unsigned int));

					selectedLeftSize = leftSize;
					selectedRightSize = rightSize;
					infoGainMax = infoGain;
					selectedThreshold = splitThreshold;

					if (!gainFound)
						gainFound = true;
				}
			}
		}
	}

	free(leftDist);
	free(rightDist);

	TreeNode *node;
	if (!gainFound)
	{
		free(selectedLeft);
		free(selectedRight);
		selectedLeft = nullptr;
		selectedRight = nullptr;
		node = nullptr;
	} else
	{
		node = new TreeNode;
		node->featureIndex = selectedFeaIndex;
		node->threshold = selectedThreshold;
		node->left = (TreeNode *) malloc(sizeof(TreeNode));
		node->right = (TreeNode *) malloc(sizeof(TreeNode));

		height++;

		bool emptyChildFound = false;

		MiniInstance *leftMiniInstanceArr = selectedMiniInstanceArr;
		node->left =
			Split(leftMiniInstanceArr, featureIndexArray, selectedLeft,
				  selectedLeftSize, height);

		MiniInstance *rightMiniInstanceArr =
			selectedMiniInstanceArr + selectedLeftSize;
		node->right =
			Split(rightMiniInstanceArr, featureIndexArray, selectedRight,
				  selectedRightSize, height);

		if (node->left == nullptr || node->right == nullptr)
			emptyChildFound = true;

		free(selectedLeft);
		free(selectedRight);
		selectedLeft = nullptr;
		selectedRight = nullptr;

		if (emptyChildFound)
			node->classIndex = getIndexOfMax(parentClassDist, numClasses);
	}

	free(selectedMiniInstanceArr);
	selectedMiniInstanceArr = nullptr;

	return node;
}

inline double TreeFactory::ComputeEntropy(const unsigned int *classDistribution,
										  const unsigned int numInstances)
{
	if (numInstances == 0)
		return 0.0;

	double entropy = 0.0;

	for (unsigned short i = 0; i < numClasses; i++)
		if (classDistribution[i] > 0 && classDistribution[i] < numInstances)
		{
			double temp = (double) classDistribution[i] / numInstances;
			entropy -= temp * log2(temp);
		}

	return entropy;
}

void TreeFactory::DestroyNode(TreeNode * node)
{
	if (node == nullptr)
		return;

	if (node->left != nullptr)
		DestroyNode(node->left);
	if (node->right != nullptr)
		DestroyNode(node->right);

	free(node->left);
	node->left = nullptr;
	free(node->right);
	node->right = nullptr;

	delete node;
	node = nullptr;
}
