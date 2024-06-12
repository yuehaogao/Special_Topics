#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <optional>
#include "al/math/al_Random.hpp"

using namespace al;
using namespace std;

#define MAX_SIZE 200
#define MAX_NUM_LAYERS 50
#define LOWERLIMIT 0.0
#define UPPERLIMIT 1.0
#define INITIAL_VALUE -1.0


// Function to generate a random float between 0.0 and 1.0
float randFloat() {
    float result = 0.0;
    for (int i = 0; i < 3; i++) {
        result += static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    return result / 3.0;  // Average of three random numbers
}


// Function to generate a n*n matrix with random floats between 0.0 and 1.0
vector<vector<float>> generateRandomMatrix(int size) {
    vector<vector<float>> matrix(size, vector<float>(size));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
    return matrix;
}


// Check if a value is in range of 0.0 - 1.0
bool inStdRange(float n) {
    if (n >= LOWERLIMIT && n <= UPPERLIMIT) {
        return true;
    } else {
        return false;
    }
}



double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}


// * Major function that determines how the hidden & output layer neurons calculate
//   - x: the relative x position in the matrix, from 0 to 1
//   - y: the relative y position in the matrix, from 0 to 1
float ANNAlgorithm(float x, float y, vector<vector<float>> input) {
    float totalSum = 0.0;
    float weight, distanceSquared;
    float N = input.size();

    // Iterate over all inputs and calculate the weighted sum
    for (int k = 0; k < N; ++k) {
        for (int l = 0; l < N; ++l) {
            distanceSquared = (x - k) * (x - k) + (y - l) * (y - l);
            weight = exp(-distanceSquared / 2.0);
            totalSum += weight * input[k][l];
        }
    }
    float rawAnswer = sigmoid(totalSum) + rnd::uniform(0.2) - 0.12;
    if (rawAnswer > 1.0) {
        rawAnswer = 1.0;
    }
    if (rawAnswer < 0.0) {
        rawAnswer = 0.0;
    }
    return pow(rawAnswer, 2);
    

}



// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// Neuron class
// Fields:
//   - Relative X Position
//   - Relative Y Position
//   - Output
class Neuron {
public:
    float posXIndex;      // Where it is in the 2D array
    float posYIndex;      // Where it is in the 2D array
    float outputValue;    // The output value, with "-1.0" meaning just initialized

    // - Constructor -
    Neuron(float posXIndexInput, float posYIndexInput) {
        // Check if the x and y positions are passed in correctly
        if (!(inStdRange(posXIndexInput)) || !(inStdRange(posYIndexInput))) {
            __throw_out_of_range("The x and y index must be within 0-1.");
        }
        // Pass in parameters
        this->posXIndex = posXIndexInput;
        this->posYIndex = posYIndexInput;
        this->outputValue = INITIAL_VALUE;
    }
 
    // Every time "onAminate" called, pass in a new matrix value
    // Each new value in the matrix must be within range of 0.0 to 1.0
    void refreshInput(vector<vector<float>> newInputValueMatrixGiven) {
        for (vector<float> oneRow : newInputValueMatrixGiven) {
            for (float oneNumber : oneRow) {
                if (!(inStdRange(oneNumber))) {
                    __throw_out_of_range("Each new value must be within 0-1.");
                }
            }
        }

        // Call the helper (but very important) function to calculate the output
        this->outputValue = ANNAlgorithm(posXIndex, posYIndex, newInputValueMatrixGiven);
    }

    float getOutput() {
        return this->outputValue;
    }
};


// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// Neuron Layer class
// Fields:
//   - Size (N)
//   - A 2D Matrix (vector of vector) of Neurons
//   - Output Matrix of float to next layer, initialized as N*N of "-1.0"
class NeuronLayer {
public:
    int size;               // The height and width of the layer
    vector<vector<Neuron>> neuronsInThisLayer;
                            // The matrix of neurons
    vector<vector<float>> outputValueMatrix;   
                            // The output values, with N*N "-1.0"s meaning just initialized

    // - Constructor -
    NeuronLayer(int sizeInput) {
        this->size = sizeInput;
        // Construct the current layer of neurons
        // Initialize the output value matrix
        for (int row = 0; row < this->size; row++) {
            vector<Neuron> newRowOfNeuron;
            vector<float> newRowOfOutput;
            for (int column = 0; column < this->size; column++) {
                float relativeX = (column + 1.0) / this->size;
                float relativeY = (row + 1.0) / this->size;
                Neuron newNeuron = Neuron(relativeX, relativeY);
                newRowOfNeuron.push_back(newNeuron);
                newRowOfOutput.push_back(newNeuron.getOutput());
            }
            this->neuronsInThisLayer.push_back(newRowOfNeuron);
            this->outputValueMatrix.push_back(newRowOfOutput);
        }
    }
 
    // Every time "onAminate" called, pass in a new matrix value
    // Each neuron is called to react by the their own "refreshInput" and "getOutput"
    // Then refresh the layer's output value matrix
    void refreshInput(vector<vector<float>> newInputValueMatrixGiven) {
        // Each new value in the matrix must be within range of 0.0 to 1.0
        // Error contracdicting it will be captured by each "Neuron" class
        // Now, check if the given matrix has a valid size
        if (newInputValueMatrixGiven.size() != this->size) {
            __throw_length_error("The height of new matrix must equal to size of ANN.");
        }
        for (vector<float> oneRow : newInputValueMatrixGiven) {
            if (oneRow.size() != this->size) {
                __throw_length_error("Each line's length of new matrix must equal to size of ANN.");
            }
        }

        // If the size is fine, call each neuron to react
        for (int row = 0; row < this->size; row++) {
            for (int column = 0; column < this->size; column++) {
                Neuron currentNeuron = this->neuronsInThisLayer[row][column];
                currentNeuron.refreshInput(newInputValueMatrixGiven);
                this->outputValueMatrix[row][column] = currentNeuron.getOutput();
            }
        }
    }

    vector<vector<float>> getOutputMatrix() {
        return this->outputValueMatrix;
    }

    void printMatrix() {
        for (vector<float> oneRow : this->getOutputMatrix()) {
            for (float oneValue : oneRow) {
                cout << oneValue << " | ";
            }
        }
        cout << endl;
    }
};



// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// All the hidden layers and the output layer
// Fields:
//   - NumLayers (total number of layers, including hidden and output, BUT NOT THE INPUT)
//   - Size (N)
//   - A 3D Cube (multiple 2D Matrix Layers) of Neurons
//   - The Final 2D Output Matrix of float, initialized as N*N of "-1.0"
//   - A 3D Archive for all neuron values
class NonInputLayers {
public:
    int numLayers;          // The total number of hidden and output layers (H + 1)
    int size;               // The height and width of each layer
    vector<NeuronLayer> nonInputNeurons;
                            // All layers of neurons
    vector<vector<float>> finalOutputValueMatrix;   
                            // The output values, with N*N "-1.0"s meaning just initialized
                            // This line decides how the "music" sounds like
    vector<vector<vector<float>>> archive;
                            // The library for all neuron's values

    // - Constructor -
    NonInputLayers(int numLayersInput, int sizeInput) {
        // Check if the parameter is valid, and does not exceed the limit.
        if (numLayersInput < 0 || numLayersInput > MAX_NUM_LAYERS) {
            __throw_out_of_range("The number of layers should be positive and cannot exceed limit.");
        }
        // Check if the parameter is valid, and does not exceed the limit.
        if (sizeInput < 0 || sizeInput > MAX_SIZE) {
            __throw_out_of_range("The input size should be positive and cannot exceed limit.");
        }
        this->numLayers = numLayersInput;
        this->size = sizeInput;

        // Construct all neurons
        // Initialize the output value matrix
        for (int layer = 0; layer <= numLayers; layer++) {
            NeuronLayer newLayer(this->size);
            this->nonInputNeurons.push_back(newLayer);
        }
        for (int row = 0; row < this->size; row++) {
            vector<float> newLine;
            for (int column = 0; column < this->size; column++) {
                newLine.push_back(INITIAL_VALUE);
            }
            this->finalOutputValueMatrix.push_back(newLine);
        }
        for (int layer = 0; layer <= this->numLayers; layer++) {
            vector<vector<float>> newMatrix;
            for (int row = 0; row < this->size; row++) {
                vector<float> newLine;
                for (int column = 0; column < this->size; column++) {
                    newLine.push_back(INITIAL_VALUE);
                }
                newMatrix.push_back(newLine);
            }
            this->archive.push_back(newMatrix);
        }
    }
 
    // Every time "onAminate" called, pass in a new matrix value
    // Each layer's neuron is called to react by the their own "refreshInput" and "getOutput"
    // Then refresh the final output value matrix
    void refreshInput(vector<vector<float>> newInputValueMatrixGiven) {
        vector<vector<vector<float>>> newArchive;
        // Size and value error will be captured by each layer and each neuron

        // Call each layer to react
        vector<vector<float>> currentInputMatrix = newInputValueMatrixGiven;
        // for (vector<float> oneLine : currentInputMatrix) {
        //     for (float one : oneLine) {
        //         cout << one << " | ";
        //     }
        // }
        // cout << endl;
        for (int currentLayer = 0; currentLayer <= this->numLayers; currentLayer++) {
            NeuronLayer currentNeuronLayer = this->nonInputNeurons[currentLayer];
            currentNeuronLayer.refreshInput(currentInputMatrix);
            currentInputMatrix = currentNeuronLayer.getOutputMatrix();
            newArchive.push_back(currentInputMatrix);
        }

        this->finalOutputValueMatrix = currentInputMatrix;
        this->archive = newArchive;
        
        
    }

    // To get the final output 2D layer's matrix value
    vector<vector<float>> getOutputMatrix() {
        return this->finalOutputValueMatrix;
    }

    // To get all the values of all neurons in each layer
    vector<vector<vector<float>>> getAllLayerOutput() {
        return this->archive;
    }

    // For testing: to print out all the layers' result
    void printAllLayerOutput() {
        int layer = 0;
        for (vector<vector<float>> oneMatrix : this->archive) {
            if (layer < this->numLayers) {
                cout << layer << "s layer: ";
            } else {
                cout << "final layer: ";
            }
            for (vector<float> oneLine : oneMatrix) {
                for (float oneValue : oneLine) {
                    cout << oneValue << " | ";
                }
            }
            cout << endl;
            layer++;
        }
        cout << endl;
    }
};
