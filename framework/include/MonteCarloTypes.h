#ifndef _MONTECARLOTYPES_H
#define _MONTECARLOTYPES_H

#include <memory>
#include <vector>
#include <iostream>
#include <cmath>

#include "game.h"
#include "Player.h"

#define ITERATION_COUNT 1000

namespace MonteCarlo {

struct TreeNode {
    Game::GameBoard boardState;
    Player::playernum_t playerNum;
    std::shared_ptr<TreeNode> parentNode = nullptr;
    std::vector<std::shared_ptr<TreeNode>> childNodes;
    double value = 0.0;
    unsigned int numTimesVisited = 0;
    unsigned int numWins = 0;
    bool simulated = false;
};

// is the node a leaf node?
static bool isLeafNode(std::shared_ptr<TreeNode> node) {
    return (node->childNodes.size() <= 0);
}

// Get Leaf Node for Selection Algorithm
// Same as getMaxNode except non-visited nodes are
// prioritized instead of ignored
static int selectLeafNode(std::vector<std::shared_ptr<TreeNode>> nodeList) {
    if (nodeList.size() <= 0) {
        std::cout << "Node List Has No Nodes!" << std::endl;
        return -1;
    }

    int maxNode = 0;

    for(int i = 0; i < nodeList.size(); ++i) {
        if(nodeList[i]->numTimesVisited == 0) {
            maxNode = i;
            break;
        }
        else if(nodeList[i]->value > nodeList[maxNode]->value) {
            maxNode = i;
        }
    }
    return maxNode;
}

// get node in list that maximizes value
static int getMaxNode(std::vector<std::shared_ptr<TreeNode>> nodeList) {
    if (nodeList.size() <= 0) {
        std::cout << "Node List Has No Nodes!" << std::endl;
        return -1;
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

// Calculates the UCT for a given node
// UCT is Upper Confidence Bound for Trees
static void calculateValue(std::shared_ptr<TreeNode> node, unsigned int rootVisits, double explorationParam) {
    double avg = ((double) node->numWins) / node->numTimesVisited;
    node->value = avg + explorationParam*sqrt(log(rootVisits) / node->numTimesVisited);
}

}

#endif