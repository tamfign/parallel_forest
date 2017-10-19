
#ifndef _UNITY_H_
#define _UNITY_H_

#include <math.h>
#include <mpi.h>
#include <algorithm>
#include <vector>

using namespace std;

#define MPI_ERROR_MESSAGE_BUFF_SIZE 50

struct Instance {
	double *featureAttrArray;
	unsigned short classIndex;
};

struct MiniInstance {
	double featureValue;
	unsigned int instanceIndex;
	unsigned short classIndex;
};

struct NumericAttr {
	char *name;
	double min;
	double max;
};

struct TreeNode {
	double threshold;
	unsigned int featureIndex;
	unsigned short classIndex;
	TreeNode *left;
	TreeNode *right;
};

bool Compare(const MiniInstance & eleX, const MiniInstance & eleY);

bool StrEqualCaseSen(const char *str1, const char *str2);
bool StrEqualCaseInsen(const char *str1, const char *str2);

unsigned int GetStrLength(const char *str);
bool IsLetter(const char c);

Instance Tokenize(const char *str, const vector < NumericAttr > &featureVec);

unsigned int getIndexOfMax(const unsigned int *uintArray,
						   const unsigned int length);

unsigned int removeDuplicates(double *sortedArr, unsigned int length);

#endif
