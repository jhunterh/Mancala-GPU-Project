#ifndef _MONTECARLOPLAYER_H
#define _MONTECARLOPLAYER_H

#include <vector>
#include <memory>

#include "MancalaStatic.h"
#include "Player.h"
#include "SimulationPolicy.h"

namespace Mancala {

struct TreeNode {
    bool isLeaf = true;
    float value = 0;
    int timesVisited = 0;
    std::shared_ptr<TreeNode> parent;
    std::vector<std::shared_ptr<TreeNode>> children;
    int playerTurn;
    std::vector<int> gameState;
} // end struct TreeNode

class MonteCarloPlayer : public Player {
public:
    MonteCarloPlayer() = default;
    ~MonteCarloPlayer() = default;

    void setSimulationPolicy(std::shared_ptr<SimulationPolicy> policy);
    int makeMove(std::vector<int> board) override;

private:

    int m_totalSimulationsRan = 0;
    std::shared_ptr<SimulationPolicy> m_simPolicy;
    std::shared_ptr<TreeNode> m_rootNode;
    std::shared_ptr<TreeNode> m_selectedNode;

    void selection();
    void expansion();
    void simulation();
    void backpropagation();

}; // end class Player

} // end namespace Mancala

#endif