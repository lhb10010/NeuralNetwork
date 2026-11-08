#include "nnet.h"
#include <cmath>
#include <atomic>
#include <thread>
#include <iostream>
#include <random>

const int DEFAULT_EPISODE_LENGTH = 10;

//
//std::thread::hardware_concurrency()


//---------------------------------------------------------------
//                    Activation Functions
//---------------------------------------------------------------   


float* softmaxActivation(float* inputLogitLayer, int logitLayerSize){

    //get largest logit
    float largestLogit = inputLogitLayer[0];
    for(int i = 0; i < logitLayerSize; i++){

        if(inputLogitLayer[i] > largestLogit){
            largestLogit = inputLogitLayer[i];
        }

    }

    float exponiatedLogits[logitLayerSize];
    float sum = 0.0;
    for(int i = 0; i < logitLayerSize; i++){
        exponiatedLogits[i] = std::exp(inputLogitLayer[i] - largestLogit);
        sum += exponiatedLogits[i];
    }

    float* finalOutput = new float[logitLayerSize];
    for(int i = 0; i < logitLayerSize; i++){

        finalOutput[i] = exponiatedLogits[i] / sum;

    }

    return finalOutput;

}


//---------------------------------------------------------------
//                  Generate Starting Values
//---------------------------------------------------------------   


float** generateStartingWeights(int firstLayerSize, int secondLayerSize){

    float** finalmatrix = new float*[firstLayerSize];

    for(int i = 0; i < firstLayerSize; i++){

        float* tempRow = new float[secondLayerSize];

        for(int j = 0; j < secondLayerSize; j++){

            tempRow[j] = 0.5;

        }

        finalmatrix[i] = tempRow;

    }

    return finalmatrix;

}


float** generateStartingBias(int firstLayerSize, int secondLayerSize){

    float** finalmatrix = new float*[firstLayerSize];

    for(int i = 0; i < firstLayerSize; i++){

        float* tempRow = new float[secondLayerSize];

        for(int j = 0; j < secondLayerSize; j++){

            tempRow[j] = 0.0;

        }

        finalmatrix[i] = tempRow;

    }

    return finalmatrix;

}


//---------------------------------------------------------------
//                          Constructor
//---------------------------------------------------------------     


nnet::nnet(int layerCount, int* layerSizes){

    this->layerCount = layerCount;
    this->layerSizes = layerSizes;
    this->activationFunction = softmaxActivation;
    this->weightsMatrices = new float**[layerCount - 1];
    this->biasMatrices = new float**[layerCount - 1];

    for(int i = 0; i < layerCount - 1; i++){

        this->weightsMatrices[i] = generateStartingWeights(layerSizes[i], layerSizes[i + 1]);

    }

    for(int i = 0; i < layerCount - 1; i++){

        this->biasMatrices[i] = generateStartingBias(layerSizes[i], layerSizes[i + 1]);

    }

    this->timesteps = new timestep[DEFAULT_EPISODE_LENGTH];
    this->timesteps->oneHotChosen = new float[this->layerSizes[this->layerCount - 1]];

}


//---------------------------------------------------------------
//                        Forwards Prop
//---------------------------------------------------------------       


const float* nnet::forwardsPropNoParallel(float* inputLayer){

    //apply activation to input and add it to outputs

    this->timesteps[timeStepPos].layerOutputs = new float*[layerCount - 1];

    float* firstInputLayer = new float[this->layerSizes[0]];
    for(int i = 0; i < this->layerSizes[0]; i++){

        firstInputLayer[i] = inputLayer[i];

    }

    
    float* currentOutputLayer;

    for(int i = 0; i < this->layerCount - 1; i++){

        int inputLayerSize = this->layerSizes[i];
        int outputLayerSize = this->layerSizes[i + 1];
        float** currentWeightMatrix = this->weightsMatrices[i];
        float** currentBiasMatrix = this->biasMatrices[i];
        currentOutputLayer = new float[outputLayerSize];

        for(int j = 0; j < outputLayerSize; j++){

            float neuronSum = 0.0;

            for(int k = 0; k < inputLayerSize; k++){

                neuronSum += inputLayer[k] * currentWeightMatrix[k][j] + currentBiasMatrix[k][j];

            }

            currentOutputLayer[j] = neuronSum; 

        }

        this->timesteps[timeStepPos].outputLayer = currentOutputLayer;
        currentOutputLayer = this->activationFunction(currentOutputLayer, outputLayerSize);
        inputLayer = currentOutputLayer;
        this->timesteps[timeStepPos].layerOutputs[i] = currentOutputLayer;

    }

    this->timeStepPos++;
    this->timesteps[timeStepPos].inputLayer = firstInputLayer;

    return currentOutputLayer;

}


//---------------------------------------------------------------
//                  Parallel Forwards Prop
//---------------------------------------------------------------    

//not usuable rn


void parallelComputeLogits(float* inputLayer, float** weights, float** bias, float* outputLogits, int portionLength, int portionPos, int inputLayerLength){


    for(int i = 0; i < portionLength; i++){

        float neuronSum = 0;

        for(int j = 0; j < inputLayerLength; j++){

            //std::cout << portionPos + i << ": " << inputLayer[j] << ", " << weights[j][portionPos + i] << ", " << bias[j][portionPos + i] << "\n";
            neuronSum += (inputLayer[j] * weights[j][portionPos + i]) + bias[j][portionPos + i];

        }

        outputLogits[portionPos + i] = neuronSum;

    }


}


float* nnet::parallelForwardsProp(float* inputLayer, int threads){

 

    float* currentOutputLayer;

    for(int i = 0; i < this->layerCount - 1; i++){

        int inputLayerSize = this->layerSizes[i];
        int outputLayerSize = this->layerSizes[i + 1];
        float** currentWeightMatrix = this->weightsMatrices[i];
        float** currentBiasMatrix = this->biasMatrices[i];
        currentOutputLayer = new float[outputLayerSize];

        int neuronsPerThread = outputLayerSize / threads;
        //std::cout << "neurons per thread: " << neuronsPerThread << "\n"; 
        std::thread* threadsArray[threads];

        for(int j = 0; j < threads; j++){

            int* currentPos = new int(neuronsPerThread * j);

            std::thread* parallelLogit = new std::thread(parallelComputeLogits, std::ref(inputLayer), std::ref(currentWeightMatrix), std::ref(currentBiasMatrix), 
                std::ref(currentOutputLayer), std::ref(neuronsPerThread), std::ref(*currentPos), std::ref(inputLayerSize));
            
            threadsArray[j] = parallelLogit;
            
        }

        for(int j = 0; j < threads; j++){

            threadsArray[j][0].join();

        }

        currentOutputLayer = this->activationFunction(currentOutputLayer, outputLayerSize);
        inputLayer = currentOutputLayer;

    }

    return currentOutputLayer;

}


//---------------------------------------------------------------
//              Set Reward For Last Forward Prop
//--------------------------------------------------------------- 


void nnet::setLastReward(float reward){

    this->timesteps[this->timeStepPos].reward = reward;

}


//---------------------------------------------------------------
//                        Choose Action
//--------------------------------------------------------------- 

int nnet::chooseAction(float* outputProbabilities){

    std::srand(static_cast<unsigned int>(std::time(0)));
    float randomFloat = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

    int i = 0;
    float sum = 0;
    int action = 0; 
    float* actions = new float[this->layerSizes[this->layerCount - 1]];
    while(true){

        sum += outputProbabilities[i];
        if(sum < randomFloat){
            action = i;
            actions[i] = 1.0;
            i++;
            break;
        }
        else{
            actions[i] = 0.0;
        }
        i++;

    }
    for(int j = i; j < this->layerSizes[this->layerCount - 1]; j++){
        actions[i] = 0.0;
    }

    this->timesteps[this->timeStepPos].oneHotChosen = actions;
    return action;

}


//---------------------------------------------------------------
//                         REINFORCE
//--------------------------------------------------------------- 


/*
void nnet::REINFORCE(){

    for(int i = 0; i < timeStepPos + 1; i++){

        float* logs = new float[this->layerSizes[this->layerCount - 1]];
        for(int j = 0; j < this->layerSizes[this->layerCount - 1]; j++){

            logs[j] = std::log(this->timesteps[0].outputLayer[j]) - this->timesteps[i].oneHotChosen[j];

        }
        for(int )

    }


}




void nnet::REINFORCE(){


    //first layer
    float* logs = new float[this->layerSizes[this->layerCount - 1]];
    for(int i = 0; i < this->layerSizes[this->layerCount - 1]; i++){

        logs[i] = std::log(this->timesteps[0].outputLayer[i]) - this->timesteps[0].oneHotChosen[i];

    }
    float** currentWeights = this->weightsMatrices[this->layerCount - 2];

    for(int i = 0; i < this->layerSizes[this->layerCount - 1]; i++){

        for(int j = 0; j < this->layerSizes[this->layerCount - 2]; j++){

            currentWeights[j][i] = currentWeights[j][i] + (this->learningRate * this->timesteps[0].reward * logs[i]);

        }

    }

    //next layers
    for(int i = this->layerCount - 2; i > 0; i++){





    }



}

*/


void nnet::REINFORCE(){

    //calculate first "error terms"
    //or whatever its called for REINFORCE
    float* errorTerms = new float[this->layerSizes[this->layerCount - 1]];
    for(int j = 0; j < this->layerSizes[this->layerCount - 1]; j++){

        errorTerms[j] = std::log(this->timesteps[0].outputLayer[j]) - this->timesteps[0].oneHotChosen[j];

    }

    //main backprop loop
    for(int i = layerCount - 1; i > -1; i--){

        float** currentWeights = this->weightsMatrices[i];
        float** currentBiases = this->biasMatrices[i];

        //calculate next error terms 
        if(i != 0){
            float* nextErrorTerms = new float[layerSizes[i]];
            for(int j = 0; j < layerSizes[i]; j++){

                //calc derivation
                nextErrorTerms[j] = 0;
                float currentNeuronRawLogit = this->timesteps[0].layerOutputs[i - 1][j];

                float maxlogit = 0.0;
                for(int k = 0; k < layerSizes[i]; k++){

                    if(this->timesteps[0].layerOutputs[i][k] > maxlogit){
                        maxlogit = this->timesteps[0].layerOutputs[i][k];
                    }
                }
                float sum = 0;
                for(int k = 0; k < layerSizes[i]; k++){
                    sum += std::exp(this->timesteps[0].layerOutputs[i][k] - maxlogit);
                }
                float currentNeuronSoftmaxDerivitive = std::exp(currentNeuronRawLogit) / sum;


                for(int k = 0; k < layerSizes[i]; k++){

                    nextErrorTerms[j] += errorTerms[k] * currentWeights[j][k] * sum;

                }

            }
        }

        //set new weights and biases
        for(int j = 0; j < layerSizes[i]; j++){

            for(int k = 0; k < layerSizes[i - 1]; k++){

                currentWeights[k][j] = this->learningRate * errorTerms[j] * this->timesteps[0].layerOutputs[i - 1][k];
                currentBiases[k][j] = this->learningRate * errorTerms[j]; 

            }

        }

    }






}





