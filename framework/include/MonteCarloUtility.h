#ifndef __MONTE_CARLO_UTILITY_H
#define __MONTE_CARLO_UTILITY_H

#include "GameTypes.h"
#include "GameBoard.h"

#define DEFAULT_BLOCK_SIZE 1024 // max thread count per block
#define DEFAULT_GRID_SIZE 2 // 2 blocks of 1024 threads

#define LAUNCH_SIZE (BLOCK_SIZE * GRID_SIZE)

typedef unsigned int gpu_count_t;
struct gpu_result
{
    gpu_count_t winCount[2] = { 0, 0 };
    gpu_count_t playCount = 0;
    gpu_count_t numMovesSimulated = 0;
};

struct deterministic_data
{
    bool isPreDetermined = false;
    unsigned int value = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

void initCuda();

int getCudaLaunchSize();

void simulationGPU(
    gpu_result* gpu_result_out,
    Game::GameBoard gameBoard,
    Player::playernum_t playerTurn,
    deterministic_data deterministicDataHost
);

#ifdef __cplusplus
}
#endif

#endif
