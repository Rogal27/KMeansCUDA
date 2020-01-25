// KMeans.Cpp/Logic.cpp
#include "Logic.h"

int KMeans::Cpp::Logic::Get() const
{
    return 42; // Really, what else did you expect?
}

int KMeans::Cpp::Logic::Sum(int tab[], int length) const
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		sum += tab[i];
	}
	return sum;
}