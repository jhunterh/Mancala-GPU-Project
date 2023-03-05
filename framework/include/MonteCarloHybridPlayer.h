#ifndef _MONTECARLOHYBRIDPLAYER_H
#define _MONTECARLOHYBRIDPLAYER_H

#include <vector>

#include "MonteCarloPlayer.h"
#include "MonteCarloTypes.h"
#include <curand_kernel.h>

#define EXPLORATION_PARAM_HYBRID 1
#define ITERATION_COUNT_HYBRID 500

typedef unsigned int gpu_count_t;

namespace Player {

// Definition of Monte Carlo Player
// This player selects a move based on the Monte Carlo Tree Search Algorithm
class MonteCarloHybridPlayer : public MonteCarloPlayer {
public:
    MonteCarloHybridPlayer();
    ~MonteCarloHybridPlayer();

    player_t getPlayerType() override { return 3; }
	std::string getDescription() override { return "Monte Carlo Hybrid Player"; }

protected:
    void runSearch() override;
    void simulation() override;
    void backpropagation() override;

private:
    void cudaSearchInit();

    curandStateMtgp32* devMTGPStates;
    mtgp32_kernel_params* devKernelParams;
    gpu_count_t* gpu_result_dev;
};

}

#endif