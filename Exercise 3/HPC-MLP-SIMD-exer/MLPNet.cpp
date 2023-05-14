#include "MLPNet.h"
#include "MLPMath.h"

typedef std::vector<std::vector<float>> Matrix2D;


MLPNet::MLPNet(const std::vector<int>& MLPTopology) {

    topology_ = MLPTopology;
    numLayers_ = (int)topology_.size() - 1;

    activationType_ = 0; // use 0 for tanh activation

    // There are numLayers of weights and biases
    // weights[0] is the weight matrix from the input layer to the first hidden layer 

    // Initialize weights and their gradients using topology
    weights_.resize(numLayers_);
    weightGradients_.resize(numLayers_);
    for (std::size_t i = 0; i < numLayers_; i++) {
        weights_[i].resize(topology_[i + 1]);
        weightGradients_[i].resize(topology_[i + 1]);
        for (std::size_t j = 0; j < topology_[i + 1]; j++) {
            weights_[i][j].resize(topology_[i]);
            weightGradients_[i][j].resize(topology_[i]);
            for (std::size_t k = 0; k < topology_[i]; k++) {
                weights_[i][j][k] = MLPMath::CalculateRandomWeight( topology_[i], topology_[i + 1]); 
                weightGradients_[i][j][k] = 0.0f;
            }
        }
    }

    // Initialize Biases and their gradients
    biases_.resize(numLayers_);
    biasGradients_.resize(numLayers_);
    for (std::size_t i = 0; i < numLayers_; i++) {
        biases_[i].resize(topology_[i + 1]);
        biasGradients_[i].resize(topology_[i + 1]);
        for (int j = 0; j < topology_[i + 1]; j++) {
            biases_[i][j] = 0.f;
            biasGradients_[i][j] = 0.f;
        }
    }

}

void MLPNet::startTraining(int numTraining, int batchSize, int numEpochs) {
    
    batchSize_ = batchSize;
    numTraining_ = numTraining;
    nBatches_ = numTraining / (batchSize);  // this must be an integer
    batchSizeV_ = batchSize_ / fvecLen;  // batchSize must be multiple of fvecLen

    // initialize storage 
    activations_.resize(batchSizeV_); // stores values at neurons after activation function has been applied
    rawOutputs_.resize(batchSizeV_); // stores neuron values before activation
    featureGradients_.resize(batchSizeV_);  // store gradients at neurons. Used in backpropagation
    for (std::size_t i = 0; i < batchSizeV_; i++) {
        // Note : Indexing of the above quantities is shifted by one from that of weights/biases
        // i.e. rawOutputs[0] is the input to the neural network, and 
        // activations[numLayers] is the prediction of the network
        activations_[i].resize(numLayers_ + 1);
        rawOutputs_[i].resize(numLayers_ + 1);
        featureGradients_[i].resize(numLayers_ + 1);
        for (std::size_t j = 0; j < (numLayers_ + 1); j++) {
            activations_[i][j].resize(topology_[j]);
            rawOutputs_[i][j].resize(topology_[j]);
            featureGradients_[i][j].resize(topology_[j]);
        }
    }

    loadMNISTEpochData(); // load all data used in training

    for (int epochIndex = 0; epochIndex < numEpochs; epochIndex++) {
        analyzeEpoch();        
    }

    printStatistics();
}

void MLPNet::analyzeEpoch() {

    for (int batchIndex = 0; batchIndex < nBatches_; batchIndex++) {
        loadBatch(batchIndex);
        analyzeBatch();
    }

    updateStatistics(); // store loss and accuracy info

    resetStatistics(); // reset counters used to calc accuracy

}

void MLPNet::loadBatch(int batchIndex) {

    // clear previous batch data
    batchClasses_.clear();
    batchInputs_.clear();

    loadBatchData(batchIndex); 
}


void MLPNet::analyzeBatch() {
    feedForward();

    analyzeOutput();

    calculateLoss();

    if (!inTesting_) {
        
        backPropagation();
        
        parameterUpdate();

        // reset gradients and feature Gradients
        resetGradients();
    }
    
}

void MLPNet::startTesting(int numTesting) {

    inTesting_ = true;

    nBatches_ = numTesting / 4;
    batchSize_ = 4;
    batchSizeV_ = 1;

    // initialize storage 
    activations_.resize(1);
    rawOutputs_.resize(1);
    featureGradients_.resize(1);  // store gradients of neurons. used in backprop
    for (std::size_t i = 0; i < 1; i++) {
        activations_[i].resize(numLayers_ + 1);
        rawOutputs_[i].resize(numLayers_ + 1);
        featureGradients_[i].resize(numLayers_ + 1);
        for (std::size_t j = 0; j < (numLayers_ + 1); j++) {
            activations_[i][j].resize(topology_[j]);
            rawOutputs_[i][j].resize(topology_[j]);
            featureGradients_[i][j].resize(topology_[j]);
        }
    }

    loadMNISTEpochData(); 

    analyzeEpoch();

}

MLPNet::~MLPNet() {

}


void MLPNet::analyzeOutput() {

    for (std::size_t subBatchIndex = 0; subBatchIndex < batchSizeV_; subBatchIndex++){
        for (std::size_t iv = 0; iv < fvecLen; iv++) {
            if (batchClasses_[subBatchIndex*fvecLen + iv] == getClassification(subBatchIndex, iv)) {
                nCorrectClassifiedEpoch_++;
            }
            else {
                nFalseClassifiedEpoch_++;
            }
        }        
    }

}


void MLPNet::feedForward() {


    for (std::size_t subBatchIndex = 0; subBatchIndex < batchSizeV_; subBatchIndex++){ // loop over batches
        // set input layer from data
        int id = subBatchIndex * fvecLen;
        for (std::size_t i = 0; i < topology_[0]; i++) {
            rawOutputs_[subBatchIndex][0][i] = fvec(batchInputs_[id][i], batchInputs_[id + 1][i],
                batchInputs_[id + 2][i], batchInputs_[id + 3][i]);
        }

        for (int layerIndex = 0; layerIndex < numLayers_; layerIndex++) { // propagate through all layers

            Matrix2D& weight = weights_[layerIndex];
            std::vector<float>& bias = biases_[layerIndex];
                
            // affine transform
            if (layerIndex == 0) { // first layer input is from data
                // Write SIMD-ized affine transform which will be used like below
                rawOutputs_[subBatchIndex][layerIndex + 1] = MLPMath::affineTransform(weight, rawOutputs_[subBatchIndex][0], bias);
            }
            else {
                rawOutputs_[subBatchIndex][layerIndex + 1] = MLPMath::affineTransform(weight, activations_[subBatchIndex][layerIndex], bias);
            }

            // apply activation function on the rawOutputs
            if (layerIndex == (numLayers_ - 1)) { // softmax on last layer
                // Write SIMD-ized softmax which will be used like below
                activations_[subBatchIndex][numLayers_] = MLPMath::applySoftmax(rawOutputs_[subBatchIndex][numLayers_]);
            }
            else {
                activations_[subBatchIndex][layerIndex + 1] = applyActivation(rawOutputs_[subBatchIndex][layerIndex+1]);
            }        
        }
    }

}


void MLPNet::backPropagation() {

    // output delta stores gradient from loss function to neurons of last layer
    std::vector<fvec> outputDelta;

    for (std::size_t subBatchIndex = 0; subBatchIndex < batchSizeV_; subBatchIndex++){

        outputDelta.clear();
        outputDelta.resize(topology_[numLayers_]);

        //calc output delta for crossEntropy + softmax combined
        for (std::size_t iv = 0; iv < fvecLen; iv++) {
            for (std::size_t i = 0; i < topology_[numLayers_]; i++) {
                outputDelta[i][iv] = activations_[subBatchIndex][numLayers_][i][iv];
                if (i == batchClasses_[subBatchIndex*fvecLen + iv]) {
                    outputDelta[i][iv] -= 1.0f;
                }
            }
        }

        // set last layer feature Gradient
        featureGradients_[subBatchIndex][numLayers_] = outputDelta;

        // propagate back through layers
        for (int layerIndex = (numLayers_ - 1); layerIndex >= 0; layerIndex--) {
            Matrix2D& weight = weights_[layerIndex];
            Matrix2D weightT = MLPMath::Transpose2D(weight);
            Matrix2D& weightGradient = weightGradients_[layerIndex];

            // backprop through activation
            // Note : backprop through last layer already handled in output delta calc.
            if (layerIndex != (numLayers_ - 1)) {
                backPropActivation(layerIndex, activationType_, subBatchIndex); // updates featureGradients_
            }
            
            // fetch gradients of the layer on the right. This will already have gradients filled
            std::vector<fvec>& rDelta = featureGradients_[subBatchIndex][layerIndex + 1];
            // calc. weight gradient for current layer
            calcWeightGradient(weight, weightGradient, rDelta, subBatchIndex, layerIndex); 
            // calc. bias gradient for current layer
            biasGradients_[layerIndex] = MLPMath::addVector(biasGradients_[layerIndex], rDelta);
            // calc. feature Gradients for current layer. Will be used as rDelta in next iteration
            featureGradients_[subBatchIndex][layerIndex] = calclFeatureGradient(weightT, rDelta);
            
        }
    }

}


void MLPNet::calcWeightGradient(Matrix2D& weight, Matrix2D& weightGradient, std::vector<fvec>& rDelta, int subBatchIndex, int layerIndex) {

    std::vector<fvec>& lactivation = activations_[subBatchIndex][layerIndex];
    if (layerIndex == 0) { // there is no activation on first layer
        lactivation = rawOutputs_[subBatchIndex][layerIndex];
    }

    fvec temp = fvec(0.0f);
    for (std::size_t i = 0; i < weightGradient.size(); i++) {
        for (std::size_t j = 0; j < weightGradient[0].size(); j++) {
            temp = rDelta[i] * lactivation[j];
            for (std::size_t iv = 0; iv < fvecLen; iv++) {
                weightGradient[i][j] += temp[iv];
            }
        }
    }

}


std::vector<float> MLPNet::calclFeatureGradient(Matrix2D& weightT, std::vector<float>& rDelta) {

    std::vector<float> lFeatureGradient = MLPMath::MatMul2D1D(weightT, rDelta);

    return lFeatureGradient;
}

std::vector<fvec> MLPNet::calclFeatureGradient(Matrix2D& weightT, std::vector<fvec>& rDelta) {

    std::vector<fvec> lFeatureGradient = MLPMath::MatMul2D1D(weightT, rDelta);

    return lFeatureGradient;
}

std::vector<float> MLPNet::applyActivation(std::vector<float>& input) {
    
    std::vector<float> output(input.size());
    switch (activationType_) {
    case 0: // TanH
        output = MLPMath::applyTanH(input);
        return output;
        break;
    case 1: // ReLU
        for (std::size_t i = 0; i < input.size(); i++) {
            if (input[i] > 0) output[i] = input[i];
            else output[i] = 0.0f;
        }
        return output;
        break;
        default: {
            return output;
            break;
        }
    }
}

std::vector<fvec> MLPNet::applyActivation(std::vector<fvec>& input) {

    std::vector<fvec> output(input.size());
    switch (activationType_) {
        case 0: {// TanH
            output = MLPMath::applyTanH(input);
            return output;
            break;
        }
        case 1: {// SIMD-ized ReLU
            output = MLPMath::applyReLU(input);
            return output;
            break;
        }
        default: {
            return output;
            break;
        }
    }
}

void MLPNet::backPropActivation(int layerIndex, int activationType, int subBatchIndex) {

    std::vector<fvec>& rDelta = featureGradients_[subBatchIndex][layerIndex + 1]; 
    std::vector<fvec>& activation = activations_[subBatchIndex][layerIndex + 1]; 
    std::vector<fvec>& rawOutput = rawOutputs_[subBatchIndex][layerIndex + 1]; 

    switch (activationType) {
        case 0: { // TanH
            fvec ones = fvec(1.0f);
            for (std::size_t i = 0; i < rDelta.size(); i++) {
                rDelta[i] = (ones - activation[i] * activation[i]) * rDelta[i];
            }
            break;
        }
        case 1:{
            fvec zero = fvec(0.0f);
            for (std::size_t i = 0; i < rDelta.size(); i++) {
                rDelta[i] = (rawOutput[i] > zero) * rDelta[i];
            }
            break;
        }
        default:{
            break;
        }
    }
}


int MLPNet::getClassification(int subBatchIndex, int iv) {
    // we use softmax activation on last layer which means activations[numLayers] gives probabilities 
    int argMax = 0;
    float max = activations_[subBatchIndex][numLayers_][0][iv];

    // find highest activation
    for (std::size_t index = 1; index < topology_[numLayers_]; index++) {
        if (activations_[subBatchIndex][numLayers_][index][iv] > max) {
            max = activations_[subBatchIndex][numLayers_][index][iv];
            argMax = index;
        }
    }
    return argMax;
}


void MLPNet::calculateLoss() {

    float output = 0.0f;
    fvec temp = fvec(0.0f);

    for (std::size_t subBatchIndex = 0; subBatchIndex < batchSizeV_; subBatchIndex++){
        for (std::size_t iv = 0; iv < fvecLen; iv++) {
            int targetClass = batchClasses_[subBatchIndex*fvecLen + iv];
            output = activations_[subBatchIndex][numLayers_][targetClass][iv];
            loss_ += -log(output);
            //std::cout << output << std::endl;

            if (isinf(loss_)) loss_ = 100.f;
        }
        //std::cout<<"loss done loop no. "<< subBatchIndex << std::endl;
    }
    //std::cout << loss_ << std::endl;
}

void MLPNet::parameterUpdate() {

    for (int layerIndex = 0; layerIndex < numLayers_; layerIndex++) {

        // update weight
        Matrix2D& oldWeight = weights_[layerIndex];
        Matrix2D& weightGradient = weightGradients_[layerIndex];
        SGD2D(oldWeight, weightGradient);

        // update bias
        std::vector<float>& oldbias = biases_[layerIndex];
        std::vector<float>& biasGradient = biasGradients_[layerIndex];
        SGD1D(oldbias, biasGradient);
    }

}

void MLPNet::resetGradients() {

    // reset weight gradients to zero
    for (std::size_t i = 0; i < weightGradients_.size(); i++) {
        for (std::size_t j = 0; j < weightGradients_[i].size(); j++) {
            for (std::size_t k = 0; k < weightGradients_[i][j].size(); k++) {
                weightGradients_[i][j][k] = 0.0f;
            }
        }
    }

    // reset bias gradients to zero
    for (std::size_t i = 0; i < biasGradients_.size(); i++) {
        for (std::size_t j = 0; j < biasGradients_[i].size(); j++) {
            biasGradients_[i][j] = 0.0f;
        }
    }

    // reset feature Gradients
    for (std::size_t i = 0; i < featureGradients_.size(); i++){
        for (std::size_t j = 0; j < featureGradients_[i].size(); j++) {
            for (std::size_t k = 0; k < featureGradients_[i][j].size(); k++) {
            featureGradients_[i][j][k] = fvec(0.0f);
            }
        }
    }

}


void MLPNet::loadMNISTEpochData() { 

    std::string filePath;
    // have to change file path to local 
    if (inTesting_){
        filePath = "AddPathToTestFileHere/mnist_test.csv";
    }
    else{
        filePath = "AddPathToTrainFileHere/mnist_train.csv";
    }

    std::ifstream file(filePath);
    if (!file) {
        std::cerr << "Could not read MNIST training data from file :" << filePath << std::endl;
    }
    // MNIST data set is 28*28 (total 784 pixels) black and white images of numbers 0-9
    // csv files store the image class(0-9) followed by pixel values(784 pixels). There are 50000 training examples

    // load required number of image data
    EpochInputs_.resize(numTraining_);
    for (std::size_t i = 0; i < EpochInputs_.size(); i++) {
        EpochInputs_[i].resize(784);
    }
    EpochClasses_.resize(numTraining_);

    std::string line, num;
    int i = 0;
    while (i < numTraining_) {

        std::getline(file, line);

        std::stringstream str(line);

        for (int lineIndex = 0; lineIndex < 785; lineIndex++) {
            std::getline(str, num, ',');
            if (lineIndex == 0) EpochClasses_[i] = std::stoi(num); // first number is image class. from 0-9
            else {
                EpochInputs_[i][lineIndex - 1] = std::stoi(num); // pixel values
            }
        }
        i++;
    }
}

void MLPNet::loadBatchData(int batchIndex) { 

    batchInputs_.resize(batchSize_);
    for (std::size_t i = 0; i < batchSize_; i++) {
        batchInputs_[i].resize(784);
    }
    batchClasses_.resize(batchSize_);

    // load examples into batch
    for (std::size_t i = 0; i < batchSize_; i++) {
        batchInputs_[i] = EpochInputs_[batchIndex * batchSize_ + i];
        batchClasses_[i] = EpochClasses_[batchIndex * batchSize_ + i];
    }
}

void MLPNet::updateStatistics() {

    int trainingSize = nCorrectClassifiedEpoch_ + nFalseClassifiedEpoch_; // numTraining_
    float accThisEpoch = (float)nCorrectClassifiedEpoch_ / (float)trainingSize;

    if (!inTesting_) { // training
        accuracyTrainingEpoch_.push_back(accThisEpoch);

        // loss per training example this epoch
        float loss = getLoss() / (float)trainingSize;
        lossTrainingEpoch_.push_back(loss);
    }
    else { // testing
        accuracyValidationEpoch_.push_back(accThisEpoch);

        // loss per validation example this epoch
        float loss = getLoss() / (float)trainingSize;
        lossValidationEpoch_.push_back(loss);
    }

    std::cout << "Correct = " << nCorrectClassifiedEpoch_ << " , Wrong =  " << nFalseClassifiedEpoch_ << std::endl;

}

void MLPNet::printStatistics() {

    if (!inTesting_) {
        for (int i = 0; i < (int)accuracyTrainingEpoch_.size(); i++) {
            std::cout << "Training Accuracy in Epoch " << i + 1 << " = " << accuracyTrainingEpoch_[i] * 100 << std::endl;
//            std::cout << "Loss in Epoch " << i + 1 << " = " << lossTrainingEpoch_[i] << std::endl;
        }
    }
    else {
        for (int i = 0; i < (int)accuracyValidationEpoch_.size(); i++) {
            std::cout << "Testing Accuracy in Epoch " << i + 1 << " = " << accuracyValidationEpoch_[i] * 100 << std::endl;
//            std::cout << "Loss in Epoch " << i + 1 << " = " << lossValidationEpoch_[i] << std::endl;
        }
    }

    //std::cout << "Correct = " << nCorrectClassifiedEpoch_ << " .Wrong =  " << nFalseClassifiedEpoch_ << std::endl;

}


void MLPNet::resetStatistics() {
    nCorrectClassifiedEpoch_ = 0;
    nFalseClassifiedEpoch_ = 0;
    loss_ = 0.f;
}

void MLPNet::SGD1D(std::vector<float>& oldWeights, std::vector<float>& gradient) {

    for (std::size_t i = 0; i < oldWeights.size(); i++) {
        oldWeights[i] -= SGDLearningRate_ * gradient[i];
    }
}

void MLPNet::SGD2D(std::vector<std::vector<float>>& oldWeights, std::vector<std::vector<float>>& gradient) {

    for (std::size_t i = 0; i < oldWeights.size(); i++) {
        for (std::size_t j = 0; j < oldWeights[0].size(); j++) {
            oldWeights[i][j] -= SGDLearningRate_ * gradient[i][j];
        }
    }
}

