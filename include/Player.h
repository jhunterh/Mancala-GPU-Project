#ifndef _PLAYER_H
#define _PLAYER_H

#include "MancalaTypes.h"

namespace Mancala {

class Player {
public:
    Player() = default;
    ~Player() = default;

    void setPlayerNumber(int newPlayerNumber) {
        m_playerNumber = newPlayerNumber;
    } // end method setPlayerNumber

    int getPlayerNumber() {
        return m_playerNumber;
    } // end method getPlayerNumber

    virtual int makeMove(MancalaBoard board){return -1;}  // function to override

private:
    int m_playerNumber = -1;

}; // end class Player

} // end namespace Mancala

#endif