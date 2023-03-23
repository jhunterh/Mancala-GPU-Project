#ifndef _MONTECARLOPLAYERMT_H
#define _MONTECARLOPLAYERMT_H

#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <functional>

#include "MonteCarloPlayer.h"

#define NUM_END_STATES_DESIRED 16
#define MAX_NUM_THREADS 4

namespace Player {

// Definition of Monte Carlo Player - Multi-Threaded
// This player selects a move based on the Monte Carlo Tree Search Algorithm
class MonteCarloPlayerMT : public MonteCarloPlayer {
public:
    MonteCarloPlayerMT();
    ~MonteCarloPlayerMT();

    std::string getDescription() override { return "Monte Carlo Player Multi-Threaded"; }
    player_t getPlayerType() override { return 2; }

    // unit testing interface
    unsigned int simulation() override;

private:
    void simulationThread();
    std::vector<std::thread> m_threads;
    std::atomic<unsigned int> m_endStatesFound;
    std::atomic<unsigned int> m_winStatesFound;
    std::atomic<unsigned int> m_waitingThreads;
    std::atomic<unsigned int> m_numMovesSimulated;
    std::atomic<bool> m_simulationDoneFlag;
    std::atomic<bool> m_gameFinishFlag;
    std::mutex m_simulationMutex;
    std::condition_variable m_simulationCondition;
};

}

#endif