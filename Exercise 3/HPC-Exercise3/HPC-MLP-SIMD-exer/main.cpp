#include <iostream>
#include <vector>
#include "MLPNet.h"

int main() {

    // ask user to input topology?
	std::vector<int> MLPTopology_ = {784, 800, 10};

    std::cout << "MLP topology : [ " << std::endl;
    for (std::size_t i = 0; i < MLPTopology_.size() - 1; i++) {
        if (i == MLPTopology_.size() - 2) {
            std::cout << "( " << MLPTopology_[i] << " , " << MLPTopology_[i + 1] << " ) ]." << std::endl;
        }
        else {
            std::cout << "( " << MLPTopology_[i] << " , " << MLPTopology_[i + 1] << " )," << std::endl;
        }
    }

    MLPNet MLPNet_ = MLPNet(MLPTopology_);

    int numTraining, batchSize, numEpochs = 0;
    std::cout << "Number of Training examples : ";
    std::cin >> numTraining;    
    std::cout << "Batch Size : "; // choose such that numTraining/batchSize is integer
    std::cin >> batchSize;
    std::cout << "Number of Epochs : ";
    std::cin >> numEpochs;

    MLPNet_.startTraining(numTraining, batchSize, numEpochs);

    int numTesting = 0;
    std::cout << "Number of Testing examples ( 0 to skip testing) : ";
    std::cin >> numTesting;
    if (numTesting > 0) {
        MLPNet_.startTesting(numTesting);
    }


}
