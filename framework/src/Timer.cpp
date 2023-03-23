#include "Timer.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>


class Timer::Implementation {
public:
    Implementation() {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    } // end default constructor

    ~Implementation() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    } // end destructor

    void start() {
        cudaEventRecord(m_start);
    } // end method start

    void stop() {
        cudaEventRecord(m_stop);
    } // end method stop

    float elapsedTime_ms() {
        cudaEventSynchronize(m_stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, m_start, m_stop);
        return milliseconds;
    } // end method elapsedTime_ms

private:
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
}; // end class Implementation

Timer::Timer() : implementation_(new Implementation()) {

} // end default constructor

void Timer::start() {
    implementation_->start();
} // end method start

void Timer::stop() {
    implementation_->stop();
} // end method stop

float Timer::elapsedTime_ms() {
    return implementation_->elapsedTime_ms();
} // end method elapsedTime_ms