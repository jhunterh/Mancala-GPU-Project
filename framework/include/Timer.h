#ifndef _TIMER_H
#define _TIMER_H

#include <memory>

class Timer {
public:
    Timer();
    void start();
    void stop();
    float elapsedTime_ms();
private:
    class Implementation;
    std::shared_ptr<Implementation> implementation_;
}; // end class

#endif