
#ifndef _ARFF_IMPORTER_H_
#define _ARFF_IMPORTER_H_

#include "Unity.h"
#include <stdio.h>
#include <string.h>

class ArffImporter {
#define READ_LINE_MAX     5000
#define TOKEN_LENGTH_MAX  35

#define KEYWORD_ATTRIBUTE "@ATTRIBUTE"
#define KEYWORD_DATA      "@DATA"
#define KEYWORD_NUMERIC   "NUMERIC"

  public:
	ArffImporter(const char *fileName);
	~ArffImporter();

	 vector < char *>GetClassAttr();
	 vector < NumericAttr > GetFeatures();
	Instance *GetInstances();
	unsigned int GetNumInstances();

  private:
	void Read(const char *fileName);
	void BuildInstanceTable();

	 vector < char *>classVec;
	 vector < NumericAttr > featureVec;
	 vector < Instance > instanceVec;

	Instance *instanceTable = nullptr;
	double *instanceBuff = nullptr;

	unsigned int numFeatures = 0;
	unsigned int numInstances = 0;
	unsigned short numClasses = 0;
};

#endif
