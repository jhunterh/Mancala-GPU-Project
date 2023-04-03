#ifndef _MONTECARLOHYBRIDPLAYER_H
#define _MONTECARLOHYBRIDPLAYER_H

#include <vector>

#include "MonteCarloPlayer.h"
#include "MonteCarloTypes.h"
#include "MonteCarloUtility.h"

namespace Player {

// Definition of Monte Carlo Player
// This player selects a move based on the Monte Carlo Tree Search Algorithm
class MonteCarloHybridPlayer : public MonteCarloPlayer {
public:
    MonteCarloHybridPlayer();
    ~MonteCarloHybridPlayer() = default;

    player_t getPlayerType() override { return 3; }
	std::string getDescription() override { return "Monte Carlo Hybrid Player"; }

    // unit testing interface
    void setDeterministic(bool isPreDetermined, int value) override;
    unsigned int simulation() override;

private:
    deterministic_data m_deterministicData;
};

}

#endif