
#ifndef _RANDOM_FOREST_H_
#define _RANDOM_FOREST_H_

#include "TreeFactory.h"
#include <omp.h>
#include <time.h>

class RandomForest {
#define RANDOM_FEATURE_SET_SIZE 10
#define NUM_TREES               100

  public:
	RandomForest();
	~RandomForest();

	void Train(const Instance * instanceTable,
			   const vector < NumericAttr > &fv,
			   const vector < char *>&cv, const unsigned int numInstances);
	void Classify(const Instance * instanceTable,
				  const unsigned int numInstances);

  private:
	 inline void Analysis(unsigned int *votes, const Instance * instanceTable,
						  const unsigned short numClasses,
						  const unsigned int correctCounter,
						  const unsigned int numInstances);
	inline void Classify(const Instance & instance, unsigned int *votes,
						 const unsigned int index);

	 vector < char *>classVec;
	 vector < NumericAttr > featureVec;

	TreeFactory treeFactory;
	TreeNode **root = nullptr;
	unsigned int numTrees;

	int rank;
	int size;
	int inited;
};

#endif
