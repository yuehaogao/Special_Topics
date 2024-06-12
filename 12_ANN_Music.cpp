#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include "ANN.cpp"


float randPosNegFloat() {
    return (randFloat() * 2.0) - 1.0;
}

int main() {
    // Seed the random number generator to ensure different results each time
    srand(time(NULL));

    // for (int i = 0; i < 10; i++) {
    //     // Print the result as a floating-point number
    //     printf("Random float between 0.0 and 1.0: %f\n", randPosNegFloat());
    // }


    vector<vector<float>> emptyMatrix;
    vector<vector<float>> m_good;
    vector<vector<float>> m_great;
    vector<vector<float>> m_bad;
    vector<vector<float>> m_soso;
    vector<vector<float>> m_super;

    vector<float> m_good_l1;
    vector<float> m_good_l2;
    m_good_l1.push_back(0.7);
    m_good_l1.push_back(0.4);
    m_good_l2.push_back(0.9);
    m_good_l2.push_back(1.0);
    m_good.push_back(m_good_l1);
    m_good.push_back(m_good_l2);

    vector<float> m_great_l1;
    vector<float> m_great_l2;
    m_great_l1.push_back(0.1);
    m_great_l1.push_back(0.7);
    m_great_l2.push_back(0.3);
    m_great_l2.push_back(0.5);
    m_great.push_back(m_great_l1);
    m_great.push_back(m_great_l2);


    vector<float> m_bad_l1;
    vector<float> m_bad_l2;
    m_bad_l1.push_back(0.5);
    m_bad_l1.push_back(0.6);
    m_bad_l2.push_back(-0.2);
    m_bad_l2.push_back(1.3);
    m_bad.push_back(m_bad_l1);
    m_bad.push_back(m_bad_l2);

    vector<float> m_soso_l1;
    vector<float> m_soso_l2;
    m_soso_l1.push_back(0.5);
    m_soso_l1.push_back(0.6);
    m_soso_l1.push_back(0.7);
    m_soso_l2.push_back(0.7);
    m_soso_l2.push_back(0.6);
    m_soso_l2.push_back(0.5);
    m_soso.push_back(m_soso_l1);
    m_soso.push_back(m_soso_l2);


    vector<float> m_super_l1;
    vector<float> m_super_l2;
    vector<float> m_super_l3;
    m_super_l1.push_back(0.1);
    m_super_l1.push_back(0.4);
    m_super_l1.push_back(0.7);
    m_super_l2.push_back(0.2);
    m_super_l2.push_back(0.5);
    m_super_l2.push_back(0.8);
    m_super_l3.push_back(0.3);
    m_super_l3.push_back(0.6);
    m_super_l3.push_back(0.9);
    m_super.push_back(m_super_l1);
    m_super.push_back(m_super_l2);
    m_super.push_back(m_super_l3);

    // InputNeuron testInputNeuron1;
    // cout << testInputNeuron1.getOutput() << endl;
    // testInputNeuron1.refreshInput(0.5);
    // cout << testInputNeuron1.getOutput() << endl;
    // testInputNeuron1.refreshInput(0.7);
    // cout << testInputNeuron1.getOutput() << endl;

    
    cout << " ---- Single Neuron Test ----" << endl;
    Neuron testHiddenNeuron(0.2, 0.9);
    cout << testHiddenNeuron.getOutput() << endl;
    testHiddenNeuron.refreshInput(emptyMatrix);
    cout << "empty: " << testHiddenNeuron.getOutput() << endl;
    testHiddenNeuron.refreshInput(m_good);
    cout << "good: " << testHiddenNeuron.getOutput() << endl;
    //testHiddenNeuron.refreshInput(m_bad);
    //cout << "bad: " << testHiddenNeuron.getOutput() << endl;

    cout << "----" << endl;

    cout << " ---- Single Layer Test ----" << endl;
    NeuronLayer testLayer(2);
    cout << "shown size: " << testLayer.size << endl;
    cout << "actual size: " << testLayer.neuronsInThisLayer.size() << endl;
    for (vector<float> oneLine : testLayer.getOutputMatrix()) {
        for (float one : oneLine) {
            cout << one << " | ";
        }
    }
    cout << endl;
    cout << "refreshed with good" << endl;
    testLayer.refreshInput(m_great);
    int refreshed_outputSize = 0;
    for (vector<float> oneLine : testLayer.getOutputMatrix()) {
        for (float one : oneLine) {
            cout << one << " | ";
            refreshed_outputSize ++;
        }
    }
    cout << "new output size: " << refreshed_outputSize << endl;
    cout << endl;
    cout << "refreshed with bad value" << endl;

    
    cout << "----" << endl;

    cout << " ---- ALL LAYER TEST ----" << endl;

    NonInputLayers littleANN = NonInputLayers(3, 3);
    cout << "construction success " << endl;
    for (vector<float> oneLine : littleANN.getOutputMatrix()) {
        for (float one : oneLine) {
            cout << one << " | ";
        }
    }

    cout << endl;
    cout << endl;
    littleANN.refreshInput(m_super);
    cout << " ---- successfully refrehed with super 3 * 3 --" << endl;

    cout << "now printing each layer's output matrix: " << endl;
    // Inspect each layer
    littleANN.printAllLayerOutput();


    cout << "Final layer: ";
    for (vector<float> oneLine : littleANN.getOutputMatrix()) {
        for (float one : oneLine) {
            cout << one << " | ";
        }
    }


    cout << endl;




    return 0;
}
