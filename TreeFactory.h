
#ifndef _TREE_FACTORY_H_
#define _TREE_FACTORY_H_

#include "Unity.h"

#include <stdio.h>
#include <string.h>

class TreeFactory {
#define MIN_NODE_SIZE          1
#define MIN_NODE_SIZE_TO_SPLIT 2

  public:
	TreeFactory(const vector < NumericAttr > &fv,
			  const vector < char *>&cv,
			  const Instance * it, const unsigned int numInstances);
	~TreeFactory();

	TreeNode *Generate(const unsigned int numFeaToSelect);
	void PrintTree(const TreeNode * node, unsigned int h);
	void DestroyNode(TreeNode * node);

  private:
	 TreeNode * Split(MiniInstance * miniInstanceArr,
					  unsigned int *featureIndexArray,
					  const unsigned int *parentClassDist,
					  const unsigned int numInstances, unsigned int height);

	inline double ComputeEntropy(const unsigned int *classDistribution,
								 const unsigned int numInstances);

	 vector < char *>classVec;
	 vector < NumericAttr > featureVec;
	const Instance *instanceTable;

	unsigned int numFeaToTry;
	unsigned int numFeaTotal;
	unsigned int numInstTotal;
	unsigned short numClasses;
};

#endif
