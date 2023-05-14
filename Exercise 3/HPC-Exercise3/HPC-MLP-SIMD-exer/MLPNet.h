#ifndef MLPNET_H
#define MLPNET_H

#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "p4_f32vec4.h"

typedef std::vector<std::vector<float>> Matrix2D;
typedef std::vector<std::vector<fvec>> Matrix2D_v;


class MLPNet {
public:

    MLPNet(const std::vector<int>& MLPTopology);

    MLPNet() { // default empty constructor
    }

    void analyzeOutput();

    int getClassification(int subBatchIndex, int iv);

    void feedForward();

    void backPropagation();

    void resetGradients();

    void calculateLoss();

    void calcWeightGradient(Matrix2D& weight, Matrix2D& weightGradient, std::vector<fvec>& rDelta, int subBatchIndex, int layerIndex);

    std::vector<float> calclFeatureGradient(Matrix2D& weight, std::vector<float>& rDelta);

    std::vector<fvec> calclFeatureGradient(Matrix2D& weight, std::vector<fvec>& rDelta);

    void parameterUpdate();

    void startTraining(int numTraining, int batchSize, int numEpochs);

    void analyzeEpoch();

    void analyzeBatch();

    void startTesting(int numTesting);

    std::vector<float> applyActivation(std::vector<float>& input);

    std::vector<fvec> applyActivation(std::vector<fvec>& input);

    void backPropActivation(int layerIndex, int activationType, int subBatchIndex);

    void SGD1D(std::vector<float>& oldWeights, std::vector<float>& gradient);

    void SGD2D(std::vector<std::vector<float>>& oldWeights, std::vector<std::vector<float>>& gradient);

    void resetStatistics();

    void printStatistics();

    void updateStatistics();

    void loadBatch(int batchIndex);

    void loadBatchData(int batchIndex);

    void loadMNISTEpochData();


    float getLoss() {
        return loss_;
    }

    ~MLPNet();

private:

    bool inTesting_ = false;
    int activationType_ = 0;

    float loss_ = 0.0f;

    std::vector<int> topology_;
    int numLayers_ = -1;
    int nBatches_ = 0;
    int batchSize_ = 0;
    int batchSizeV_ = 0;
    int numTraining_ = 0;
    int numValidation_ = 0;

    std::vector<Matrix2D> weights_;
    std::vector<Matrix2D> weightGradients_;
    Matrix2D biases_;
    Matrix2D biasGradients_;

    Matrix2D batchInputs_;
    std::vector<int> batchClasses_;
    Matrix2D EpochInputs_;
    std::vector<int> EpochClasses_;

    // storing 
    std::vector<Matrix2D_v> featureGradients_;
    // store raw output, rawOutput[subBatchIndex][0] is input
    std::vector<Matrix2D_v> rawOutputs_; 
    // store activation (after applying non-linearity)
    std::vector<Matrix2D_v> activations_;  // input is activations[subBatchIndex][0] and output is activations[subBatchIndex][last]

    float SGDLearningRate_ = 0.001f;

    // storing results
    std::vector<float> accuracyTrainingEpoch_;
    std::vector<float> lossTrainingEpoch_;
    std::vector<float> accuracyValidationEpoch_;
    std::vector<float> lossValidationEpoch_;


    // Counter for statistics:
    int nCorrectClassifiedEpoch_ = 0;
    int nFalseClassifiedEpoch_ = 0;

};

#endif // MLPNET_H
