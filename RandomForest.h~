
#ifndef _RANDOM_FOREST_H_
#define _RANDOM_FOREST_H_

#include "TreeBuilder.h"
#include <omp.h>
#include <time.h>


class RandomForest
{
#define RANDOM_FEATURE_SET_SIZE 10
#define NUM_TREES               100

public:
    RandomForest();
    ~RandomForest();

    void Train(
        const Instance* instanceTable,
        const vector<NumericAttr>& fv,
        const vector<char*>& cv,
        const unsigned int numInstances );
    void Classify( 
        const Instance* instanceTable,
        const unsigned int numInstances );

private:
    // Return the index of the predicted class
    inline void Classify(
        const Instance& instance,
        unsigned int* votes,
        const unsigned int index );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    
    TreeBuilder treeBuilder;
    TreeNode** root = nullptr;
    unsigned int numTrees;

    int rank;
    int size;
    int inited;
};

#endif
