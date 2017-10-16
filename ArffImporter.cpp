
#include "ArffImporter.h"


using namespace std;

ArffImporter::ArffImporter()
{
    
}

ArffImporter::~ArffImporter()
{
    free( instanceBuff );
    free( instanceTable );

    for (char* classAttr : classVec) free( classAttr );
    classVec.clear();

    for (NumericAttr& feature : featureVec) free( feature.name );
    featureVec.clear();
}

void ArffImporter::BuildInstanceTable()
{
    if (instanceBuff != nullptr || instanceTable != nullptr)
        return;
    
    instanceBuff =
        (double*) malloc( numInstances * numFeatures * sizeof( double ) );
    instanceTable = (Instance*) malloc( numInstances * sizeof( Instance ) );
    for (unsigned int i = 0; i < numInstances; i++)
    {
        double* offset = instanceBuff + i * numFeatures;
        memmove(
            offset,
            instanceVec[i].featureAttrArray,
            numFeatures * sizeof( double ) );
        instanceTable[i].featureAttrArray = offset;
        instanceTable[i].classIndex = instanceVec[i].classIndex;
        free( instanceVec[i].featureAttrArray );
    }

    instanceVec.clear();
}

// Need to check string length boundary
void ArffImporter::Read( const char* fileName )
{
    FILE *fp;

    if ((fp = fopen( fileName, "r+" )) == nullptr)
	{
		printf( "File: %s not found!\n", fileName );
		return;
	}

    // Assuming all data types of all features are double
    // and ignoring feature types
    char firstToken[TOKEN_LENGTH_MAX];
    char buffer[READ_LINE_MAX];

    while (fgets( buffer, READ_LINE_MAX, fp ) != nullptr)
    {
        // Skip empty lines
        if (buffer[0] == '\n') continue;

        int readSize;
        sscanf( buffer, "%s%n", firstToken, &readSize );

        if (StrEqualCaseInsen( firstToken, KEYWORD_ATTRIBUTE ))
        {
            char* featureName = (char*) malloc( TOKEN_LENGTH_MAX );
            char* featureType = (char*) malloc( TOKEN_LENGTH_MAX );
            
            sscanf( buffer + readSize, "%s %s", featureName, featureType );

            // Read feature names
            if (StrEqualCaseInsen( featureType, KEYWORD_NUMERIC ))
            {
                //printf( "Feature name: %s, length: %d \n", 
                //    featureName, GetStrLength( featureName ) );

                NumericAttr feature;
                feature.name       = featureName;
                feature.min        = 0.0;
                feature.max        = 0.0;
                feature.mean       = 0.0;
                featureVec.push_back( feature );
            }
            // Read class names
            else
            {
                // Parse classes attributes
                char* className = (char*) malloc( TOKEN_LENGTH_MAX );
                featureType++;
                
                while (sscanf( featureType, "%[^,}]%n", className, &readSize ) > 0)
                {
                    printf( "Class name: %s \n", className );

                    classVec.push_back( className );
                    className = (char*) malloc( TOKEN_LENGTH_MAX );

                    featureType += readSize + 1;
                }
            }

            continue;
        }
        // Read feature values
        else if (StrEqualCaseInsen( firstToken, KEYWORD_DATA ))
        {
            numFeatures = featureVec.size();
            numClasses = classVec.size();
            
            unsigned int featureAttrArraySize =
                numFeatures * sizeof( double );

            double* featureValueSumArr = (double*) calloc( numFeatures, 
                sizeof( double ) );

            while (fgets( buffer, READ_LINE_MAX, fp ) != nullptr)
            {
                unsigned int index = 0;
                unsigned int featureIndex = 0;
                double value;
                
                Instance instance;
                instance.featureAttrArray = (double*) malloc( featureAttrArraySize );

                // Get feature attribute value
                while (sscanf( buffer + index, "%lf%n", &value, &readSize ) > 0)
                {
                    if (featureVec[featureIndex].min > value)
                        featureVec[featureIndex].min = value;
                    
                    if (featureVec[featureIndex].max < value)
                        featureVec[featureIndex].max = value;

                    featureValueSumArr[featureIndex] += value;
                    instance.featureAttrArray[featureIndex++] = value;
                    index += readSize + 1;
                }

                // Get class attribute value
                char classValue[TOKEN_LENGTH_MAX];
                sscanf( buffer + index, "%s%n", classValue, &readSize );

                for (unsigned short i = 0; i < numClasses; i++)
                {
                    if (StrEqualCaseSen( classVec[i], classValue ))
                    {
                        instance.classIndex = i;
                        break;
                    }
                }

                instanceVec.push_back( instance );
            }

            unsigned int instanceSize = instanceVec.size();

            // Compute bucket size and mean value for each numerical attribute
            for (unsigned int i = 0; i < numFeatures; i++)
            {
                featureVec[i].mean = featureValueSumArr[i] / instanceSize;

                // printf(
                //     "feature %u, max: %f, min: %f, mean: %f\n",
                //     i,
                //     featureVec[i].max,
                //     featureVec[i].min,
                //     featureVec[i].mean );
            }

            free( featureValueSumArr );
            featureValueSumArr = nullptr;

            break;
        }
    }

    numInstances = instanceVec.size();

    fclose( fp );
    BuildInstanceTable();
}

vector<char*> ArffImporter::GetClassAttr()
{
    return classVec;
}

vector<NumericAttr> ArffImporter::GetFeatures()
{
    return featureVec;
}

Instance* ArffImporter::GetInstances()
{
    return instanceTable;
}

unsigned int ArffImporter::GetNumInstances()
{
    return numInstances;
}
