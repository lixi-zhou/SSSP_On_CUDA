#ifndef TIMER_HPP
#define TIMER_HPP
#include <time.h>

class Timer{
    private:
        time_t startTime;
        time_t stopTime;

    public:
        void start();
        void stop();
        int elapsedTime();

};
#endif