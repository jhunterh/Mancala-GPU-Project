#include "GameBoard.h"
#include <sstream>

namespace Game {

typedef uint8_t corner_t;
enum CornerType 
{
    CORNER_MIN = 0,
    CORNER_END = 4
};


CUDA_CALLABLE_MEMBER
inline static boardpos_t getCornerPos(boardpos_t currentPos, corner_t cornerIdx)
{
    static const boardpos_t cornerList[GAME_BOARD_SIZE][4] = {
        {-1, -1, 4, 5},{-1, -1, 5, 6},{-1, -1, 6, 7},{-1, -1, 7, -1},
        {-1, 0, -1, 8},{0, 1, 8, 9},{1, 2, 9, 10},{2, 3, 10, 11},
        {4, 5, 12, 13},{5, 6, 13, 14},{6, 7, 14, 15},{7, -1, 15, -1},
        {-1, 8, -1, 16},{8, 9, 16, 17},{9, 10, 17, 18},{10, 11, 18, 19},
        {12, 13, 20, 21},{13, 14, 21, 22},{14, 15, 22, 23},{15, -1, 23, -1},
        {-1, 16, -1, 24},{16, 17, 24, 25},{17, 18, 25, 26},{18, 19, 26, 27},
        {20, 21, 28, 29},{21, 22, 29, 30},{22, 23, 30, 31},{23, -1, 31, -1},
        {-1, 24, -1, -1},{24, 25, -1, -1},{25, 26, -1, -1},{26, 27, -1, -1}
    };

    return cornerList[currentPos][cornerIdx];
}

// Init the board state
void GameBoard::initBoard()
{
    boardState = boardstate_t {
        .isOccupiedBoard = 0xFFF00FFF,
        .isBlackBoard = 0x00000FFF,
        .isKingBoard = 0x00000000
    };
}

// Execute a move on the board for a given player
CUDA_CALLABLE_MEMBER
moveresult_t GameBoard::executeMove(move_t move, Player::playernum_t playerNum)
{
    // Currently unused
    moveresult_t result = MOVE_SUCCESS;

    // Verify old space is occupied and correct player
    bitboard_t oldMask = (1 << move.oldPos);
    bool isBlackPlayer = static_cast<bool>(playerNum);
    if((!(boardState.isOccupiedBoard & oldMask)) ||
        (isBlackPlayer != static_cast<bool>(boardState.isBlackBoard & oldMask)))
    {
        result = MOVE_INVALID;
    }

    // Check if jump
    bool isJump = (move.jumpPos != -1);
    bitboard_t newMask = (1 << move.newPos);
    bitboard_t jumpMask = (isJump << move.jumpPos);
    if(isJump)
    {
        // Verify move spaces are next to current
        bool isSpacesNext = false;
        for(uint8_t cornerIdx = CornerType::CORNER_MIN; 
            cornerIdx < CornerType::CORNER_END; 
            cornerIdx++)
        {
            boardpos_t jumpPosCheck = getCornerPos(move.oldPos, cornerIdx);
            boardpos_t newPosCheck = getCornerPos(jumpPosCheck, cornerIdx);
            isSpacesNext = isSpacesNext || ((jumpPosCheck == move.jumpPos) && 
                            (newPosCheck == move.newPos));
        }

        // Verify jump space is occupied and other player
        // Also Verify other space is empty
        if(((!(boardState.isOccupiedBoard & jumpMask)) || 
            (isBlackPlayer == static_cast<bool>(boardState.isBlackBoard & jumpMask)) ||
            (boardState.isOccupiedBoard & newMask)) &&
            isSpacesNext)
        {
            result = MOVE_INVALID;
        }
    }
    else
    {
        // Verify move space is next to current
        bool isSpacesNext = false;
        for(uint8_t cornerIdx = CornerType::CORNER_MIN; 
            cornerIdx < CornerType::CORNER_END; 
            cornerIdx++)
        {
            boardpos_t newPosCheck = getCornerPos(move.oldPos, cornerIdx);
            isSpacesNext = isSpacesNext || (newPosCheck == move.newPos);
        }

        // Verify new space is empty
        if((boardState.isOccupiedBoard & newMask) && isSpacesNext)
        {
            result = MOVE_INVALID;
        }
    }
    
    // Only make changes if move is valid
    if(result == MOVE_SUCCESS)
    {
        // Clear old board state
        boardState.isOccupiedBoard &= ~oldMask; 

        // Handle new space
        boardState.isOccupiedBoard |= newMask;
        if(isBlackPlayer)
            boardState.isBlackBoard |= newMask;
        else
            boardState.isBlackBoard &= ~newMask;

        // Handle king piece
        boardpos_t kingCheckPlayerOffset = playerNum*28;
        bool isKing = ((move.newPos >= kingCheckPlayerOffset) &&
                        ((move.newPos <= (kingCheckPlayerOffset + 3)))) || 
                        (boardState.isKingBoard & oldMask);

        // If already a king or make it to end of board
        if(isKing)
            boardState.isKingBoard |= newMask;
        else
            boardState.isKingBoard &= ~newMask;
        
        // Handle jump
        boardState.isOccupiedBoard &= ~jumpMask;  // Clear jump spot if jump

        // Handle move again (double jump)
        bool canJumpAgain = false;
        if(isJump)
        {
            // Double jumping backwards is always allowed
            for(uint8_t cornerIdx = CornerType::CORNER_MIN; 
                cornerIdx < CornerType::CORNER_END; 
                cornerIdx++)
            {
                boardpos_t movePos = getCornerPos(move.newPos, cornerIdx);
                if(movePos != BOARD_POS_INVALID)
                {
                    // Check if space empty or not
                    bitboard_t moveMask = (1 << movePos);
                    if(boardState.isOccupiedBoard & moveMask)
                    {
                        // Space is not empty, look for jump

                        // First make sure piece is opposite color
                        bool isJumpPieceOpposing = ((bool)(boardState.isBlackBoard & moveMask))
                                                        != isBlackPlayer;

                        // Make sure jumping space is free
                        boardpos_t jumpPos = getCornerPos(movePos, cornerIdx);
                        bitboard_t jumpMask = (1 << jumpPos);
                        bool isJumpSpaceFree = (jumpPos != BOARD_POS_INVALID) && 
                                                !(boardState.isOccupiedBoard & jumpMask);

                        // If valid, add jump
                        canJumpAgain = (isJumpPieceOpposing && isJumpSpaceFree) || canJumpAgain;
                    }
                }
            }
        }
        result += canJumpAgain;
    }

    return result;
}

// Return the possible move on the board for a given player
CUDA_CALLABLE_MEMBER
movecount_t GameBoard::getMoves(movelist_t& movesOut, Player::playernum_t playerNum)
{
    // Initialize counts and lists
    movecount_t moveCount = 0;
    movecount_t jumpCount = 0;
    movelist_t moveList;
    movelist_t jumpList;

    // Determine typical corner min and max
    const uint8_t cornerMinTypical = 2*playerNum;
    const uint8_t cornerMaxTypical = 2 + cornerMinTypical;
    // For each board square
    for(boardpos_t pos = 0; pos < GAME_BOARD_SIZE; pos++)
    {
        bitboard_t mask = (1 << pos);
        bool playerPiece = (boardState.isOccupiedBoard & mask) && 
                            (((bool)(boardState.isBlackBoard & mask)) == ((bool) playerNum));
        if(playerPiece)
        {
            // Piece is player's piece
            // Find all moves and jumps

            // For each corner
            bool isKing = boardState.isKingBoard & mask;
            const uint8_t cornerMin = isKing ? CornerType::CORNER_MIN : cornerMinTypical;
            const uint8_t cornerMax = isKing ? CornerType::CORNER_END : cornerMaxTypical;
            for(uint8_t cornerIdx = cornerMin; cornerIdx < cornerMax; cornerIdx++)
            {
                boardpos_t movePos = getCornerPos(pos, cornerIdx);
                if(movePos != BOARD_POS_INVALID)
                {
                    // Check if space empty or not
                    bitboard_t moveMask = (1 << movePos);
                    if(boardState.isOccupiedBoard & moveMask)
                    {
                        // Space is not empty, look for jump

                        // First make sure piece is opposite color
                        bool isJumpPieceOpposing = ((bool)(boardState.isBlackBoard & moveMask))
                                                     != ((bool) playerNum);

                        // Make sure jumping space is free
                        boardpos_t jumpPos = getCornerPos(movePos, cornerIdx);
                        bitboard_t jumpMask = (1 << jumpPos);
                        bool isJumpSpaceFree = (jumpPos != BOARD_POS_INVALID) && 
                                                !(boardState.isOccupiedBoard & jumpMask);

                        // If valid, add jump
                        if(isJumpPieceOpposing && isJumpSpaceFree)
                        {
                            move_t jump;
                            jump.oldPos = pos;
                            jump.newPos = jumpPos;
                            jump.jumpPos = movePos;
                            jumpList[jumpCount++] = jump;
                        }
                    }
                    else
                    {
                        // Space is empty, add move
                        move_t move;
                        move.oldPos = pos;
                        move.newPos = movePos;
                        move.jumpPos = BOARD_POS_INVALID;
                        moveList[moveCount++] = move;
                    }
                }
            }
            
        }
    }

    // Return jumps list if jump exists
    const movecount_t finalMoveCount = jumpCount ? jumpCount : moveCount;
    const movelist_t* finalMoveList = jumpCount ? &jumpList : &moveList;
    memcpy(&movesOut, finalMoveList, finalMoveCount*sizeof(move_t));
    return finalMoveCount;
}

// Return the board result
CUDA_CALLABLE_MEMBER
boardresult_t GameBoard::getBoardResult(Player::playernum_t currentPlayerNum)
{
    // Get other player num
    Player::playernum_t otherPlayerNum = !currentPlayerNum;

    // Get move counts
    movelist_t moves;
    bool canCurrentPlayerMove = static_cast<bool>(getMoves(moves, currentPlayerNum));
    bool canOtherPlayerMove = static_cast<bool>(getMoves(moves, otherPlayerNum));

    // Calculate board result
    boardresult_t result = GAME_ACTIVE;
    result += (otherPlayerNum + 1)*(!canCurrentPlayerMove) + 
                (currentPlayerNum + 1)*(!canOtherPlayerMove);
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
    boardStateBuf << "-----------------------\n";
    for(uint8_t row = 0; row < 8; row++)
    {
        boardStateBuf << "|   ";
        if((row % 2) == 0)
            boardStateBuf << "  ";
        for(uint8_t col = 0; col < 4; col++)
            boardStateBuf << stateChars[row*4 + col] << "   ";
            if(row % 2)
                boardStateBuf << "  ";
        boardStateBuf << "|\n";
    }
    boardStateBuf << "-----------------------";
    
    // Return board string
    return boardStateBuf.str();
}

std::string GameBoard::getMoveString(move_t move)
{
    return std::string("Old: " + std::to_string(move.oldPos) + 
                        ", New: " + std::to_string(move.newPos) + 
                        ", Jump: " + std::to_string(move.jumpPos));
}

// Set the board to a random state
void GameBoard::scramble()
{
    //TODO: set game board to a random state
}

}