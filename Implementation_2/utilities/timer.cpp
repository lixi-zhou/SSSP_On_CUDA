#include "timer.hpp"


void Timer::start(){
    this->startTime = clock();
}

int Timer::stop(){
    this->stopTime = clock();
    return (stopTime - startTime);
}