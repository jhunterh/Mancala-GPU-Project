#ifndef _MONTECARLOTYPES_H
#define _MONTECARLOTYPES_H

#include <memory>
#include <vector>
#include <iostream>

#include "game.h"
#include "Player.h"

namespace MonteCarlo {

struct TreeNode {
    Game::boardstate_t boardState;
    Player::playernum_t playerNum;
    std::shared_ptr<TreeNode> parentNode = nullptr;
    std::vector<std::shared_ptr<TreeNode>> childNodes;
    double value = 0.0;
    unsigned int numTimesVisited = 0;
}

static bool isLeafNode(std::shared_ptr<TreeNode> node) {
    return (node->childNodes.size() <= 0);
}

static int getMaxNode(std::vector<std::shared_ptr<TreeNode>> nodeList) {
    if (nodeList.size() <= 0) {
        std::cout << "Node List Has No Nodes!" << std::endl;
        return -1
    }

    int maxNode = 0;

    for(int i = 0; i < nodeList.size(); ++i) {
        if(nodeList[i]->numTimesVisited == 0) {
            continue;
        }
        else if(nodeList[i]->value > nodeList[maxNode]->value) {
            maxNode = i;
        }
    }
    return maxNode;
}

}

#endif