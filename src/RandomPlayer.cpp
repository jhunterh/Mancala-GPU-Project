#include <ctime>
#include <cstdlib>
#include <iostream>

#include "RandomPlayer.h"

namespace Mancala {

    RandomPlayer::RandomPlayer() {
        srand(time(NULL));
    } // end default constructor

    int RandomPlayer::makeMove(std::vector<int> board) {

        int playerNum = getPlayerNumber();

        std::vector<int> validMoves = getValidMoves(board, playerNum);

        int randomValue = (rand() % validMoves.size());

        return validMoves[randomValue];
    } // end method makeMove

} // end namespace Mancala