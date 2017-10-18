
#ifndef _TREE_FACTORY_H_
#define _TREE_FACTORY_H_

#include "BasicDataStructures.h"
#include "Helper.h"

#include <stdio.h>
#include <string.h>


class TreeFactory
{
#define MIN_NODE_SIZE          1
#define MIN_NODE_SIZE_TO_SPLIT 2
#define NUM_CHILDREN           2

public:
    TreeFactory();
    ~TreeFactory();

    void Init(
        const vector<NumericAttr>& fv,
        const vector<char*>& cv,
        const Instance* it,
        const unsigned int numInstances );
    TreeNode* Generate( const unsigned int numFeaToSelect );
    void PrintTree( const TreeNode* node, unsigned int h );
    void DestroyNode( TreeNode* node );

private:
    TreeNode* Split(
        MiniInstance* miniInstanceArr,
        unsigned int* featureIndexArray,
        const unsigned int* parentClassDist,
        const unsigned int numInstances,
        unsigned int height );
    inline double ComputeGini(
        const unsigned int* classDistribution,
        const unsigned int numInstances );
    inline double ComputeEntropy(
        const unsigned int* classDistribution,
        const unsigned int numInstances );

    vector<char*> classVec;
    vector<NumericAttr> featureVec;
    const Instance* instanceTable;

    unsigned int numFeaToTry;
    unsigned int numFeaTotal;
    unsigned int numInstTotal;
    unsigned short numClasses;
};

#endif
