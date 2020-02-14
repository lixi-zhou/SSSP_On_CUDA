#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include <iostream>
#include <stdio.h>
#include <string>


using namespace std;

typedef unsigned int uint;

const unsigned int MAX_DIST = 65535;


void printDist(uint* dist, uint size);

void printProcessed(bool* processed, uint size);

void printPreNode(uint* preNode, uint size);

#endif