#ifndef _RANDOMPLAYER_H
#define _RANDOMPLAYER_H

#include <iostream>
#include <vector>

#include "Player.h"

namespace Player {

// Definition of Random Player
// This player always selects a move at random
class RandomPlayer : public Player {
public:
    RandomPlayer();
    ~RandomPlayer() = default;

    player_t getPlayerType() override { return 0; }
	std::string getDescription() override { return "Random Player"; }
    std::string getPerformanceDataString() override {return (getDescription()+": No Performance Data\n");}
	Game::move_t selectMove(Game::GameBoard& board, playernum_t playerNum);

    // unit testing interface
    void setDeterministic(bool isPreDetermined, int value)
    {
        m_isPreDetermined = isPreDetermined;
        if(m_isPreDetermined)
        {
            m_preDeterminedValue = value;
        }
    }

private:
    bool m_isPreDetermined = false;
    int m_preDeterminedValue = 0;

};

}

#endif