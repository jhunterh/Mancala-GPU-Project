#include "GameBoard.h"

namespace Game {

    
// TODO: CAN_JUMP(oldPos, newPos) function
// TODO: CAN_MOVE(oldPos, newPos) function
// OR make jump list
// TODO: IS_VALID(oldPos, newPos) function
static const boardpos_t jumpOffsets[4] = {-9, -7, 7, 9};

// Init the board state
void GameBoard::initBoard()
{
    boardState = boardstate_t {
        .isOccupiedBoard = 0xFFF00FFF,
        .isBlackBoard = 0xFFF00000,
        .isKingBoard = 0x00000000
    }
}

// Execute a move on the board for a given player
CUDA_CALLABLE_MEMBER moveresult_t GameBoard::executeMove(move_t move, Player::playernum_t playerNum)
{
    // Currently unused
    // TODO: Add invalid move checking
    (void) playerNum;

    // Handle old space
    // TODO: Check if old space is occupied and correct player
    // TODO: Check if new space is empty
    bitboard_t oldMask = (1 << move.oldPos);
    boardState.isOccupiedBoard &= ~oldMask; // Clear old board state

    // Handle new space
    // Change to no if statements
    bitboard_t newMask = (1 << move.newPos);
    boardState.isOccupiedBoard |= newMask;
    if(boardState.isBlackBoard & newMask)
        boardState.isBlackBoard |= newMask;
    else
        boardState.isBlackBoard &= ~newMask;
    if(boardState.isKingBoard & newMask)
        boardState.isKingBoard |= newMask;
    else
        boardState.isKingBoard &= ~newMask;
    
    // Handle jump
    // TODO: Check if jump space is occupied and other player
    bool isJump = (move.jumpPos != -1);
    boardState.isOccupiedBoard &= ~(isJump << move.jumpPos);  // Clear jump spot if jump

    // Handle move again
    // TODO: Add check to do another jump
    return MoveResult::MOVE_SUCCESS;
}

// Return the possible move on the board for a given player
CUDA_CALLABLE_MEMBER movecount_t GameBoard::getMoves(movelist_t& movesOut, Player::playernum_t playerNum)
{
    // Implement code here
    movecount_t moveCount = 0;
    movecount_t jumpCount = 0;
    movelist_t moveList;
    movelist_t jumpList;
    for(boardpos_t pos = 0; pos < GAME_BOARD_SIZE; pos++)
    {
        bitboard_t mask = (1 << pos);
        bool playerPiece = (boardState.isOccupiedBoard & mask) && 
                            (((bool)(boardState.isBlackBoard & mask)) == ((bool) playerNum));
        if(playerPiece)
        {
            // Piece is player's piece
            // Find all moves and jumps

            // TODO: Cornerlist alternative?
            // TODO: Handle not being king
            for(uint8_t cornerIdx = 0; cornerIdx < 4; cornerIdx++)
            {
                boardpos_t movePos = cornerList[pos][cornerIdx];
                if(movePos != BOARD_POS_INVALID)
                {
                    // Check if space empty or not
                    bitboard_t moveMask = (1 << movePos);
                    if(boardState.isOccupiedBoard & mask)
                    {
                        // Space is not empty, look for jump

                        // First make sure piece is opposite color
                        bool isJumpPieceOpposing = ((bool)(boardState.isBlackBoard & moveMask))
                                                     != ((bool) playerNum);

                        // Make sure jumping space is free
                        boardpos_t jumpPos = cornerList[movePos][cornerIdx];
                        bitboard_t jumpMask = (1 << jumpPos);
                        bool isJumpSpaceFree = (jumpPos != BOARD_POS_INVALID) && 
                                                !(boardState.isOccupiedBoard & jumpMask);

                        // If valid, add jump
                        if(isJumpPieceOpposing && isJumpSpaceFree)
                        {
                            jumpList[jumpCount++] = move_t {
                                .oldPos = pos,
                                .newPos = movePos,
                                .jumpPos = jumpPos
                            };
                        }
                    }
                    else
                    {
                        // Space is empty, add move
                        moveList[moveCount++] = move_t {
                            .oldPos = pos,
                            .newPos = movePos,
                            .jumpPos = BOARD_POS_INVALID
                        };
                    }
                }
            }
            
        }
    }

    // Return jumps list if jump exists
    movesOut = jumpCount ? jumpList : moveList;
    return jumpCount ? jumpCount : moveCount;
}

// Return the board result
CUDA_CALLABLE_MEMBER boardresult_t GameBoard::getBoardResult(Player::playernum_t currentPlayerNum)
{
    // Get other player num
    playernum_t otherPlayerNum = !currentPlayerNum;

    // Get move counts
    movelist_t moves;
    bool canCurrentPlayerMove = getMoves(moves, currentPlayerNum);
    bool canOtherPlayerMove = getMoves(moves, otherPlayerNum);

    // Calculate board result
    boardresult_t result = GAME_ACTIVE;
    result += (currentPlayerNum + 1)*(canCurrentPlayerMove) + 
                (otherPlayerNum + 1)*(canOtherPlayerMove);
    return result;
}

// Return the state of the board in string format
std::string GameBoard::getBoardStateString()
{
    // Build list of string characters for board
    char stateChars[GAME_BOARD_SIZE];
    for(boardpos_t pos = 0; pos < GAME_BOARD_SIZE; pos++)
    {
        stateChars[pos] = '.';
        bitboard_t mask = (1 << pos);
        if(boardState.isOccupiedBoard & mask)
        {
            stateChars[pos] = 'r';
            if(boardState.isBlackBoard & mask) stateChars[pos] -= 0x10;
            if(boardState.isKingBoard & mask) stateChars[pos] -= 0x20;
        }
    }

    // Format board
    std::stringstream boardStateBuf;
    boardStateBuf << "-------------------\n";
    for(uint8_t row = 0; row < 8; row++)
    {
        boardStateBuf << "| ";
        if((row % 2) == 0)
            boardStateBuf << "  ";
        for(uint8_t col = 0; col < 4; col++)
            boardStateBuf << stateChars[row*4 + col] << " ";
        if(row % 2)
            boardStateBuf << "  ";
        boardStateBuf << "|\n";
    }
    boardStateBuf << "-------------------";
    
    // Return board string
    return boardStateBuf.str();
}

}