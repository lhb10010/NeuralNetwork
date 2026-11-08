


class nnet{

    struct timestep{

        float* inputLayer;
        float* outputLayer;
        float* oneHotChosen;
        float reward = 0;
        float** layerOutputs;

    };

    public:


        nnet(int layerCount, int* layerSizes);

        const float* forwardsPropNoParallel(float* inputLayer);
        float* parallelForwardsProp(float* inputLayer, int threads);

        void setLastReward(float reward);
        void REINFORCE();

        int chooseAction(float* outputProbabilities);


    private:


        int layerCount;
        int* layerSizes;

        float*** weightsMatrices;
        float*** biasMatrices;
        float* (*activationFunction)(float*, int); //when the class is created this variable is set to the desired activation function. 
        //This allows the activation function to be chosen dynamically for each instance of nnet

        timestep* timesteps;
        int timeStepPos = -1;

        float learningRate = 0.1;

};
