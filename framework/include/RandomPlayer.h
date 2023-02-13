#ifndef _RANDOMPLAYER_H
#define _RANDOMPLAYER_H

#include <vector>

#include "MancalaStatic.h"
#include "Player.h"

namespace Mancala {

class RandomPlayer : public Player {
public:
    RandomPlayer();
    ~RandomPlayer() = default;

    int makeMove(std::vector<int> board) override;

}; // end class Player

} // end namespace Mancala

#endif