
#include "RandomForest.h"

RandomForest::RandomForest()
{
    // Init MPI
    MPI_Initialized( &inited );
    if (!inited) MPI_Init( nullptr, nullptr );

    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
}

RandomForest::~RandomForest()
{
    // Destory trees
    if (root != nullptr)
    {
        for (unsigned int i = 0; i < numTrees; i++)
            treeBuilder.DestroyNode( rootArr[i] );
        free( root );
        root = nullptr;
    }

    MPI_Initialized( &inited );
    if (inited) MPI_Finalize();
}

void RandomForest::Train(
    const Instance* instanceTable,
    const vector<NumericAttr>& fv,
    const vector<char*>& cv,
    const unsigned int numInstances )
{
    classVec = cv;
    featureVec = fv;

    if (NUM_TREES % size > 0)
    {
        numTrees = NUM_TREES / size + 1;
        if (rank == size - 1)
            numTrees = NUM_TREES - numTrees * (size - 1);
    }
    else
        numTrees = NUM_TREES / size;

    root = (TreeNode**) malloc( numTrees * sizeof( TreeNode* ) );
    treeBuilder.Init( fv, cv, instanceTable, numInstances );

    srand( mpiNodeId + 1 );
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (unsigned int treeId = 0; treeId < numTrees; treeId++)
        {
            rootArr[treeId] = treeBuilder.BuildTree( RANDOM_FEATURE_SET_SIZE );
        }
    }
}

void RandomForest::Classify(
    const Instance* instanceTable,
    const unsigned int numInstances )
{
    unsigned short numClasses = classVec.size();
    unsigned int correctCounter = 0;
    unsigned int* votes = (unsigned int*) 
        calloc( numClasses * numInstances, sizeof( unsigned int ) );

    #pragma omp parallel for schedule(dynamic)
    for (unsigned int i = 0; i < numInstances; i++)
        Classify( instanceTable[i], votes, i );

    if (mpiNodeId == 0)
        CheckMPIErr( MPI_Reduce( MPI_IN_PLACE, votes, numClasses * numInstances, 
            MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD ), mpiNodeId );
    else
        CheckMPIErr( MPI_Reduce( votes, nullptr, numClasses * numInstances, 
            MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD ), mpiNodeId );

    if (mpiNodeId == 0)
    {
        Analysis(votes);
    }

    free( votes );
    votes = nullptr;
}

inline void RandomForest::Analysis(unsigned int* votes)
{
		#pragma omp parallel for reduction(+: correctCounter) schedule(static)
        for (unsigned int i = 0; i < numInstances; i++)
        {
            unsigned short predictedClassIndex = 
                getIndexOfMax( votes + i * numClasses, numClasses );
            if (predictedClassIndex == instanceTable[i].classIndex)
                correctCounter++;
        }

        double correctRate = (double) correctCounter / (double) numInstances;
        double incorrectRate = 1.0 - correctRate;

        printf( "Correct rate: %f\n", correctRate );
        printf( "Incorrect rate: %f\n", incorrectRate );
}

inline void RandomForest::Classify(
    const Instance& instance, 
    unsigned int* votes, 
    const unsigned int instId )
{
    unsigned short numClasses = classVec.size();

    for (unsigned int treeId = 0; treeId < numTrees; treeId++)
    {
        TreeNode* node = rootArr[treeId];
        if (node == nullptr) continue;

        while (node->childrenArr != nullptr)
        {
            unsigned int childId = (unsigned int)
                (instance.featureAttrArray[node->featureIndex] >= node->threshold);
            if (node->childrenArr[childId] == nullptr) break;
            else node = node->childrenArr[childId];
        }

        votes[instId * numClasses + node->classIndex]++;
    }
}
