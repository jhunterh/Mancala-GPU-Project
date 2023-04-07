#ifndef _MONTECARLOTYPES_H
#define _MONTECARLOTYPES_H

#include <memory>
#include <vector>
#include <iostream>
#include <cmath>

#include "game.h"
#include "Player.h"
#include "Logger.h"

namespace MonteCarlo {

struct TreeNode {
    Game::GameBoard boardState;
    Player::playernum_t playerNum;
    std::shared_ptr<TreeNode> parentNode = nullptr;
    std::vector<std::shared_ptr<TreeNode>> childNodes;
    double value = 0.0;
    unsigned int numTimesVisited = 0;
    double numWins = 0;
    bool simulated = false;
};

// compare two nodes
static bool nodeCompare(std::shared_ptr<TreeNode> nodeA, std::shared_ptr<TreeNode> nodeB) {
    bool equals = true;

    if(nodeA->boardState.getBoardStateString() != nodeB->boardState.getBoardStateString()) {
        equals = false;
    } else if(nodeA->playerNum != nodeB->playerNum) {
        equals = false;
    } else if(nodeA->value != nodeB->value) {
        equals = false;
    } else if(nodeA->numTimesVisited != nodeB->numTimesVisited) {
        equals = false;
    } else if(nodeA->numWins != nodeB->numWins) {
        equals = false;
    } else if(nodeA->simulated != nodeB->simulated) {
        equals = false;
    }

    return equals;
}

// is the node a leaf node?
static bool isLeafNode(std::shared_ptr<TreeNode> node) {
    return (node->childNodes.size() <= 0);
}

// Get Leaf Node for Selection Algorithm
// Same as getMaxNode except non-visited nodes are
// prioritized instead of ignored
static int selectLeafNode(std::vector<std::shared_ptr<TreeNode>> nodeList) {
    Logging::Logger& logger = Logging::Logger::getInstance();
    if (nodeList.size() <= 0) {
        logger.log(Logging::SIMULATION_LOG,"Node List Has No Nodes!");
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
    Logging::Logger& logger = Logging::Logger::getInstance();
    if (nodeList.size() <= 0) {
        logger.log(Logging::SIMULATION_LOG,"Node List Has No Nodes!");
        return -1;
    }

    int maxNode = 0;

    for(int i = 0; i < nodeList.size(); ++i) {
        if(nodeList[i]->numTimesVisited == 0) {
            continue;
        }
        else if(nodeList[i]->numWins > nodeList[maxNode]->numWins) {
            maxNode = i;
        }
    }

    return maxNode;
}

// Calculates the UCT for a given node
// UCT is Upper Confidence Bound for Trees
static void calculateValue(std::shared_ptr<TreeNode> node, unsigned int rootVisits, double explorationParam) {
    double avg = node->numWins / node->numTimesVisited;
    node->value = avg + (explorationParam*sqrt(log(rootVisits) / node->numTimesVisited));
}

// Report given at the end of each simulation cycle
struct SimulationPerformanceReport {
    unsigned int numMovesSimulated = 0;
    double executionTime = 0.0;
};

}

#endif