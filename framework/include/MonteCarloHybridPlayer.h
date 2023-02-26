#ifndef _MONTECARLOHYBRIDPLAYER_H
#define _MONTECARLOHYBRIDPLAYER_H

#include <vector>

#include "MonteCarloPlayer.h"
#include "MonteCarloTypes.h"

namespace Player {

// Definition of Monte Carlo Player
// This player selects a move based on the Monte Carlo Tree Search Algorithm
class MonteCarloHybridPlayer : public MonteCarloHybridPlayer {
public:
    MonteCarloHybridPlayer() = default;
    ~MonteCarloHybridPlayer() = default;

    player_t getPlayerType() override { return 3; }
	std::string getDescription() override { return "Monte Carlo Hybrid Player"; }

private:
    void simulation() override;
};

}

#endif