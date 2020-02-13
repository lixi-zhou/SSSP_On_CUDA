#include "timer.hpp"


void Timer::start(){
    this->startTime = clock();
}

void Timer::stop(){
    this->stopTime = clock();
}

int Timer::elapsedTime(){
    return (this->stopTime - this->startTime);
}