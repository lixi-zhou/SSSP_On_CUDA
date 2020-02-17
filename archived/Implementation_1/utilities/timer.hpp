#ifndef TIMER_HPP
#define TIMER_HPP
#include <time.h>
#include <chrono> // C++ 11 standard library for timing

using namespace std;

class Timer{
    private:
        chrono::steady_clock::time_point startTime;
        chrono::steady_clock::time_point stopTime;

    public:
        void start();
        void stop();
        double elapsedTime();

};
#endif