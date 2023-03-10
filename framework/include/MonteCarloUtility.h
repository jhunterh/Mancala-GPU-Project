#ifndef __MONTE_CARLO_UTILITY_H
#define __MONTE_CARLO_UTILITY_H

#include "GameTypes.h"
#include "GameBoard.h"

#define BLOCK_SIZE 128
#define PLAYCOUNT_THRESHOLD_GPU 200

// Note: For future use
// Launch configurator says 576 block size and 216 grid size for max launch
#define GRID_SIZE 216
#define LAUNCH_SIZE (BLOCK_SIZE * GRID_SIZE)

typedef unsigned int gpu_count_t;
struct gpu_result
{
    gpu_count_t winCount[2] = { 0, 0 };
    gpu_count_t playCount = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

void curandInit();

void simulationGPU(
    gpu_result* gpu_result_out,
    Game::GameBoard gameBoard,
    Player::playernum_t playerTurn
);

#ifdef __cplusplus
}
#endif

#endif
