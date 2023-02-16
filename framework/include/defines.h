#ifndef DEFINES_H
#define DEFINES_H

#include <cstdint>

namespace Player
{

typedef int8_t playernum_t;
enum
{
    PLAYER_NUMBER_INVALID = -1,
    PLAYER_NUMBER_MIN = 0,
    PLAYER_NUMBER_1 = 0,
    PLAYER_NUMBER_2 = 1,
    PLAYER_NUMBER_MAX = 1,
};

typedef uint8_t playertype_t;

};

#endif // DEFINES