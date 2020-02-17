#include "timer.hpp"


void Timer::start(){
    this->startTime = chrono::steady_clock::now();
}

void Timer::stop(){
    this->stopTime = chrono::steady_clock::now();
}

double Timer::elapsedTime(){
    double elapsedTime = (double)(chrono::duration_cast<chrono::microseconds>(this->stopTime - this->startTime).count()) / 1000;

    return elapsedTime;
}