#include "helperfuncs.h"
#include <math.h>

real recipCostDer(real val, real pol, real k)
{
	real alpha = 0.5;
	real e = exp(1);
	real x = val*pol;  //FIXME
	if(x < 0.5)
		return -k*exp(-x/alpha);
	else
		return -k/e*alpha*alpha/x/x;
}
