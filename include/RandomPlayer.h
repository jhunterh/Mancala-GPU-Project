#ifndef _RANDOMPLAYER_H
#define _RANDOMPLAYER_H

#include "MancalaTypes.h"
#include "Player.h"

namespace Mancala {

class RandomPlayer : public Player {
public:
    RandomPlayer() = default;
    ~RandomPlayer() = default;

    int makeMove(MancalaBoard board) override;

}; // end class Player

} // end namespace Mancala

#endif