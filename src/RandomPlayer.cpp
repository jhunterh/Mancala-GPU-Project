#include <ctime>
#include <cstdlib>
#include <iostream>

#include "RandomPlayer.h"
#include "MancalaTypes.h"

namespace Mancala {

    RandomPlayer::RandomPlayer() {
        srand(time(NULL));
    } // end default constructor

    int RandomPlayer::makeMove(MancalaBoard board) {

        int playerNum = getPlayerNumber();

        std::vector<int> validMoves = getValidMoves(board, playerNum);

        int randomValue = (rand() % validMoves.size());

        return validMoves[randomValue];
    } // end method makeMove

} // end namespace Mancala