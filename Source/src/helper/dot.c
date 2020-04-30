#include "helperfuncs.h"

real dot(real* a, real* b, int size)
{
	int i;
	real result=0.0;
	for(i=0; i<size; i++)
	{
		result += a[i] * b[i];
	}
    return result;
}
