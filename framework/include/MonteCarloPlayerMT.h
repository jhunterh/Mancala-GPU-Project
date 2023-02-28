#ifndef _MONTECARLOPLAYERMT_H
#define _MONTECARLOPLAYERMT_H

#include <vector>
#include <atomic>

#include "MonteCarloPlayer.h"

#define NUM_END_STATES_DESIRED 4
#define NUM_THREADS 4
#define EXPLORATION_PARAM 5
#define ITERATION_COUNT 1000

namespace Player {

// Definition of Monte Carlo Player - Multi-Threaded
// This player selects a move based on the Monte Carlo Tree Search Algorithm
class MonteCarloPlayerMT : public MonteCarloPlayer {
public:
    MonteCarloPlayerMT() = default;
    ~MonteCarloPlayerMT() = default;

    std::string getDescription() override { return "Monte Carlo Player Multi-Threaded"; }
    player_t getPlayerType() override { return 2; }

protected:
    void runSearch() override;
    void simulation() override;
    void backpropagation() override;

private:
    std::atomic<unsigned int> m_endStatesFound;
    std::atomic<unsigned int> m_winStatesFound;
    void simulationThread();
};

}

#endif