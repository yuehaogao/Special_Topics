// MAT594P, Spring 2024
// Yuehao Gao
// Designed based on Myungin Lee(2022) Sine Envelope with Visuals

// This is an Allosphere version of Artificial Neural Network (ANN)
// Being illustrated and sonified
// The initiative was inspired by the following YouTube Video: 
// https://www.youtube.com/watch?v=Tsvxx-GGlTg&t=1s

// The class consists of a ANN sturcutre: nonInputLayers "hiddenAndOutputNeurons"
// Which consists of all neurons in the hidden and the output layers
// Processing the input matrix feed to the individual input layer

// For the input layer:
// The values are initialized to 0.0
// Each time it is refreshed, it follows a "moving oval" crawling on the canvas
// https://editor.p5js.org/Yuehao_Gao/full/geABl5pWf

// For the hidden and output layers:
// Each neuron processes the values in the previous layer
// According to the "ANNAlgorithm" method in the ANN.cpp file
// The output layer is supposed to "blurrily outline" the input layer in a "Mosaic" manner
// The output layer is also sonified by:
//    - The very center is sonified to the tone of C3
//    - Follows a full-note-scale system
//    - The further the fired neuron is in the output layer, the higher its pitch

// ----------------------------------------------------------------

// * Important parameters *:
//    - REFRESH_THRESHOLD:         how many times "onAminate" called trigger the next refresh
//                                 "onAnimate" is called 60 times / second
//                                 hence a value of 60 means 1 refresh / second
//                                 and a value of 1 means 60 refresh / second
//    - ANN_SIZE:                  the width of each layer
//    - ANN_NUM_HIDDEN_LAYERS:     the number of hidden layers of the ANN
//                                 not including the output layer
//    - inputLayerFireThreshold:   how large should a number in the input layer be to
//                                 "light up" the particle
//    - firingThreshold:           how large should a number in the hidden layer be to
//                                 "light up" the particle
//    - outputLayerFireThreshold:  how large should a number in the output layer be to
//                                 "light up" the particle

// ----------------------------------------------------------------

// Press '=' to enable/disable navigation
// Press '[' or ']' to turn on & off GUI 

// ----------------------------------------------------------------




#include <cmath>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <time.h> 

#include "Gamma/Analysis.h"
#include "Gamma/Effects.h"
#include "Gamma/Envelope.h"
#include "Gamma/Oscillator.h"
#include "Gamma/DFT.h"
//#include "al/app/al_App.hpp"
#include "al/app/al_DistributedApp.hpp"
#include "al/app/al_GUIDomain.hpp"
#include "al_ext/statedistribution/al_CuttleboneDomain.hpp"
#include "al_ext/statedistribution/al_CuttleboneStateSimulationDomain.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/io/al_MIDI.hpp"
#include "al/math/al_Functions.hpp"
#include "al/scene/al_PolySynth.hpp"
#include "al/scene/al_SynthSequencer.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"

#include "ANN.cpp"   // The ANN Class defined

// using namespace gam;
using namespace al;
using namespace std;


// * ----------------------------------- *
// * ---- * IMPORTANT PARAMETERS * ----- *

#define REFRESH_THRESHOLD 30          // The refresh rate of oval
#define ANN_SIZE 25
#define ANN_NUM_HIDDEN_LAYERS 2
const float inputLayerFireThreshold = 0.33;
const float firingThreshold = 0.66;
const float outputLayerFireThreshold = 0.9;

// * ----------------------------------- *
// * ----------------------------------- *


#define FFT_SIZE 4048
#define CHANGE_MOTION_LOWER_LIMIT 180
#define CHANGE_MOTION_UPPER_LIMIT 360
#define WHITE_H 0.0
#define WHITE_S 0.0
#define WHITE_V 100.0
#define PI 3.1415926535
#define OVAL_MARCHING_SPEED 0.09
#define OVAL_LARGEST 0.3
#define OVAL_SMALLEST 0.15

const float pointSize = 0.05;
const float pointDistance = 0.3;
const float layerDistance = 7.0 * pointDistance;
const float lineWidth = 0.5;





// This example shows how to use SynthVoice and SynthManagerto create an audio
// visual synthesizer. In a class that inherits from SynthVoice you will
// define the synth's voice parameters and the sound and graphic generation
// processes in the onProcess() functions.

class SineEnv : public SynthVoice
{
public:
  // Unit generators
  gam::Pan<> mPan;
  gam::Sine<> mOsc;
  gam::Env<3> mAmpEnv;
  // envelope follower to connect audio output to graphics
  gam::EnvFollow<> mEnvFollow;
  // Draw parameters
  Mesh mMesh;
  double a;
  double b;
  double spin = al::rnd::uniformS();
  double timepose = 0;
  Vec3f note_position;
  Vec3f note_direction;

  // Additional members
  // Initialize voice. This function will only be called once per voice when
  // it is created. Voices will be reused if they are idle.
  void init() override
  {
    // Intialize envelope
    mAmpEnv.curve(0); // make segments lines
    mAmpEnv.levels(0, 1, 1, 0);
    mAmpEnv.sustainPoint(2); // Make point 2 sustain until a release is issued

    // We have the mesh be a sphere
    addSphere(mMesh, 0.3, 50, 50);
    mMesh.decompress();
    mMesh.generateNormals();

    createInternalTriggerParameter("amplitude", 0.03, 0.0, 1.0);
    createInternalTriggerParameter("frequency", 60, 20, 5000);
    createInternalTriggerParameter("attackTime", 0.66, 0.01, 3.0);
    createInternalTriggerParameter("releaseTime", 1.5, 0.1, 10.0);
    createInternalTriggerParameter("pan", 0.0, -1.0, 1.0);

    // Initalize MIDI device input
  }

  // The audio processing function
  void onProcess(AudioIOData &io) override
  {
    // Get the values from the parameters and apply them to the corresponding
    // unit generators. You could place these lines in the onTrigger() function,
    // but placing them here allows for realtime prototyping on a running
    // voice, rather than having to trigger a new voice to hear the changes.
    // Parameters will update values once per audio callback because they
    // are outside the sample processing loop.
    mOsc.freq(getInternalParameterValue("frequency"));
    mAmpEnv.lengths()[0] = getInternalParameterValue("attackTime");
    mAmpEnv.lengths()[2] = getInternalParameterValue("releaseTime");
    mPan.pos(getInternalParameterValue("pan"));
    while (io())
    {
      float s1 = mOsc() * mAmpEnv() * getInternalParameterValue("amplitude");
      float s2;
      mEnvFollow(s1);
      mPan(s1, s1, s2);
      io.out(0) += s1;
      io.out(1) += s2;
    }
    // We need to let the synth know that this voice is done
    // by calling the free(). This takes the voice out of the
    // rendering chain
    if (mAmpEnv.done() && (mEnvFollow.value() < 0.001f))
      free();
  }

  // The graphics processing function
  void onProcess(Graphics &g) override
  {}

  // The triggering functions just need to tell the envelope to start or release
  // The audio processing function checks when the envelope is done to remove
  // the voice from the processing chain.
  void onTriggerOn() override
  {
    float angle = getInternalParameterValue("frequency") / 200;
    mAmpEnv.reset();
    a = al::rnd::uniform();
    b = al::rnd::uniform();
    timepose = 0;
    note_position = {0, 0, 0};
    note_direction = {sin(angle), cos(angle), 0};
  }

  void onTriggerOff() override { mAmpEnv.release(); }
};


// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// The shared state between local and the Allosphere Terminal
struct CommonState {

  Pose pose;    
  float pointSize;
  float pointDistance;
  float layerDistance;
  float lineWidth;

  // *** 
  // These 2D and 3D arrays are causing "al_OSC.cpp" to complain

  // Neuron Points
  Vec3f inputLayerNeuronFixedPosition[ANN_SIZE][ANN_SIZE];
  HSV inputLayerNeuronRealTimeColor[ANN_SIZE][ANN_SIZE];
  Vec3f hiddenLayerNeuronFixedPosition[ANN_NUM_HIDDEN_LAYERS][ANN_SIZE][ANN_SIZE];
  HSV hiddenLayerNeuronRealTimeColor[ANN_NUM_HIDDEN_LAYERS][ANN_SIZE][ANN_SIZE];
  Vec3f outputLayerNeuronFixedPosition[ANN_SIZE][ANN_SIZE];
  HSV outputLayerNeuronRealTimeColor[ANN_SIZE][ANN_SIZE];

  // Lines
  Vec3f linesStartingFixedPosition[(ANN_NUM_HIDDEN_LAYERS + 1)][ANN_SIZE * ANN_SIZE];
  Vec3f linesEndingFixedPosition[(ANN_NUM_HIDDEN_LAYERS + 1)][ANN_SIZE * ANN_SIZE];
  HSV linesRealTimeColor[(ANN_NUM_HIDDEN_LAYERS + 1)][ANN_SIZE * ANN_SIZE];
  
  //***

};


// To slurp a file
string slurp(string fileName);

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// The main "app" structure
struct MyApp : public DistributedAppWithState<CommonState>, public MIDIMessageHandler
{
public:
  
  SynthGUIManager<SineEnv> synthManager{"SineEnv"};                                      // GUI manager for SineEnv voices
  RtMidiIn midiIn; // MIDI input carrier
  Mesh mSpectrogram;
  vector<float> spectrum;
  bool showGUI = true;
  bool showSpectro = true;
  bool navi = true;

  int frameCount;
  int frameIndex;
  

  // ------------------ OUR PRESCIOUS ANN STRUCTURE ------------------
  NonInputLayers hiddenAndOutputNeurons = NonInputLayers(ANN_NUM_HIDDEN_LAYERS, ANN_SIZE);  // The hidden neurons that producing the data output
  vector<vector<float>> liveInputMatrix;                                                 // The live input matrix
  //vector<vector<vector<bool>>> isNeuronFired;                                            // The live referee tracking if each neurion is "lit up"
  // -----------------------------------------------------------------

  // Parameters for the oval
  float canvasSize;
  float ovalCenterX;
  float ovalCenterY;
  float ovalDX;
  float ovalDY;
  float ovalRadiusX;
  float ovalRadiusY;
  float ovalSizeChangeDirectionX;
  float ovalSizeChangeDirectionY;
  float ovalNextChangeTimeX;
  float ovalNextChangeTimeY;

  // List of triggered MIDI Notes
  vector<int> MIDINoteTriggeredLastTime;

  // STFT variables
  gam::STFT stft = gam::STFT(FFT_SIZE, FFT_SIZE / 4, 0, gam::HANN, gam::MAG_FREQ);

  // Shader and meshes
  ShaderProgram pointShader;

  Mesh InputLayer;
  Mesh HiddenLayers;
  Mesh OutputLayer;
  Mesh ConnectionLines;
  

  // --------------------------------------------------------
  // onCreate
  void onCreate() override {
    bool createPointShaderSuccess = pointShader.compile(slurp("../point_tools/point-vertex.glsl"),
                                                        slurp("../point_tools/point-fragment.glsl"),
                                                        slurp("../point_tools/point-geometry.glsl"));

    if (!createPointShaderSuccess) {
      exit(1);
    }

    // Set up the parameters for the oval
    frameCount = 0;
    frameIndex = 0;
    canvasSize = ANN_SIZE * pointDistance;
    ovalCenterX = 0.0;
    ovalCenterY = 0.0;
    ovalRadiusX = 0.2 * canvasSize;
    ovalRadiusY = 0.2 * canvasSize;
    float angle = rnd::uniform(2.0 * PI);
    float speed = OVAL_MARCHING_SPEED;
    ovalDX = speed * cos(angle);
    ovalDY = speed * sin(angle);
    ovalSizeChangeDirectionX = rnd::uniform(2.0) - 1.0;
    ovalSizeChangeDirectionY = rnd::uniform(2.0) - 1.0;
    ovalNextChangeTimeX = CHANGE_MOTION_LOWER_LIMIT + rnd::uniform(CHANGE_MOTION_UPPER_LIMIT - CHANGE_MOTION_LOWER_LIMIT);
    ovalNextChangeTimeY = CHANGE_MOTION_LOWER_LIMIT + rnd::uniform(CHANGE_MOTION_UPPER_LIMIT - CHANGE_MOTION_LOWER_LIMIT);


    // Initialize the input values and the neuron activity referee
    vector<vector<float>> initialLiveInputMatrix(ANN_SIZE, vector<float>(ANN_SIZE, 0.0f));
    liveInputMatrix = initialLiveInputMatrix;
    // vector<vector<vector<bool>>> initialIsNeuronFired((ANN_NUM_HIDDEN_LAYERS + 1), vector<vector<bool>>(ANN_SIZE, vector<bool>(ANN_SIZE, false)));
    // isNeuronFired = initialIsNeuronFired;

    // ------------------------------------------------------------
    // Initialize parameters for all meshes

    InputLayer.primitive(Mesh::POINTS);
    HiddenLayers.primitive(Mesh::POINTS);
    OutputLayer.primitive(Mesh::POINTS);
    ConnectionLines.primitive(Mesh::LINES);

    
    // The input layer
    for (int ipRow = 0; ipRow < ANN_SIZE; ipRow++) {
      for (int ipColumn = 0; ipColumn < ANN_SIZE; ipColumn++) {
        float x = (ipRow - (ANN_SIZE * 0.5)) * pointDistance;
        float y = (ipColumn - (ANN_SIZE * 0.5)) * pointDistance;
        float z = (ANN_NUM_HIDDEN_LAYERS + 1) * layerDistance;

        // addCube(InputLayer);
        // Mat4f xfm;
        // xfm.setIdentity;
        // xfm.scale(Vec3f(pointSize, pointSize, pointSize));
        // xfm.translate(Vec3f(x, y, z));
        // InputLayer.transform(xfm, InputLayer.vertices().size());

        InputLayer.vertex(Vec3f(x, y, z));
        state().inputLayerNeuronFixedPosition[ipRow][ipColumn] = Vec3f(x, y, z);
        InputLayer.color(HSV(0.0f, 0.0f, 0.7f));    // Input layer initialized as white
        state().inputLayerNeuronRealTimeColor[ipRow][ipColumn] = HSV(0.0f, 0.0f, 0.7f);
        InputLayer.texCoord(1.0, 0);
      }
    }

    // The hidden layers
    for (int hdLayer = 0; hdLayer < ANN_NUM_HIDDEN_LAYERS; hdLayer++) {
      for (int hdRow = 0; hdRow < ANN_SIZE; hdRow++) {
        for (int hdColumn = 0; hdColumn < ANN_SIZE; hdColumn++) {
          float x = (hdRow - (ANN_SIZE * 0.5)) * pointDistance;
          float y = (hdColumn - (ANN_SIZE * 0.5)) * pointDistance;
          float z = ((ANN_NUM_HIDDEN_LAYERS + 1) * layerDistance) - ((hdLayer + 1) * layerDistance) + rnd::uniform(3 * pointDistance) - 1.5 * pointDistance;

          HiddenLayers.vertex(Vec3f(x, y, z));
          state().hiddenLayerNeuronFixedPosition[hdLayer][hdRow][hdColumn] = Vec3f(x, y, z);
          HiddenLayers.color(HSV(0.0f, 1.0f, 0.3f));  // Input layer initialized as dark red
          state().hiddenLayerNeuronRealTimeColor[hdLayer][hdRow][hdColumn] = HSV(0.0f, 1.0f, 0.3f);
          HiddenLayers.texCoord(1.0, 0);
        }
      }
    }

    // The output layer
    for (int opRow = 0; opRow < ANN_SIZE; opRow++) {
      for (int opColumn = 0; opColumn < ANN_SIZE; opColumn++) {
        float x = (opRow - (ANN_SIZE * 0.5)) * pointDistance;
        float y = (opColumn - (ANN_SIZE * 0.5)) * pointDistance;
        float z = 0.0;

        OutputLayer.vertex(Vec3f(x, y, z));
        state().outputLayerNeuronFixedPosition[opRow][opColumn] = Vec3f(x, y, z);
        OutputLayer.color(HSV(0.0f, 0.0f, 0.7f));  // Input layer initialized as white
        state().inputLayerNeuronRealTimeColor[opRow][opColumn] = HSV(0.0f, 0.0f, 0.7f);
        OutputLayer.texCoord(1.0, 0);
      }
    }
    
    
    
    // ------------------------------------------------------------

    navControl().active(false); // Disable navigation via keyboard, since we
                                // will be using keyboard for note triggering
    // Set sampling rate for Gamma objects from app's audio
    gam::sampleRate(audioIO().framesPerSecond());
    imguiInit();
    // Play example sequence. Comment this line to start from scratch
    synthManager.synthRecorder().verbose(true);

    if (isPrimary()) {
      nav().pos(-35.0, 0.0, 40.0);
      nav().faceToward(0.0, 0.0, 2.2);
    }
  }


  // --------------------------------------------------------
  // onInit
  void onInit() override {
    // Try starting the program. If not successful, exit.
    auto cuttleboneDomain =
        CuttleboneStateSimulationDomain<CommonState>::enableCuttlebone(this);
    if (!cuttleboneDomain) {
      std::cerr << "ERROR: Could not start Cuttlebone. Quitting." << std::endl;
      quit();
    }

    if (isPrimary()) {
      // Check for connected MIDI devices
      if (midiIn.getPortCount() > 0)
      {
        // Bind ourself to the RtMidiIn object, to have the onMidiMessage()
        // callback called whenever a MIDI message is received
        MIDIMessageHandler::bindTo(midiIn);

        // Set up GUI
        // auto GUIdomain = GUIDomain::enableGUI(defaultWindowDomain());
        // auto& gui = GUIdomain->newGUI();

        // Open the last device found
        unsigned int port = midiIn.getPortCount() - 1;
        midiIn.openPort(port);
        printf("Opened port to %s\n", midiIn.getPortName(port).c_str());
      } else {
        printf("Actually, no MIDI devices found, please use Keyboard.\n");
      }
      // Declare the size of the spectrum 
      spectrum.resize(FFT_SIZE / 2 + 1);
    }
  }

  // --------------------------------------------------------
  // onSound
  // The audio callback function. Called when audio hardware requires data
  void onSound(AudioIOData &io) override
  {
    synthManager.render(io); // Render audio
    // STFT
    while (io())
    {
      if (stft(io.out(0)))
      { // Loop through all the frequency bins
        for (unsigned k = 0; k < stft.numBins(); ++k)
        {
          // Here we simply scale the complex sample
          spectrum[k] = 10.0 * tanh(pow(stft.bin(k).real(), 1.5) );
          //spectrum[k] = stft.bin(k).real();
        }
      }
    }
  }


  // --------------------------------------------------------
  // onAnimate
  void onAnimate(double dt) override
  {
    // The GUI is prepared here
    imguiBeginFrame();
    frameIndex += 1;
   
    if (frameCount >= REFRESH_THRESHOLD) {
      frameCount = 0;

      // Step (1): stop the MIDI notes in the previous turn
      for (int oneNote : MIDINoteTriggeredLastTime) {
        synthManager.triggerOff(oneNote);
      }
      vector<int> emptyIntegerList;
      MIDINoteTriggeredLastTime = emptyIntegerList;

      // Step (2): refresh the input value and color according to the movement of the circle -------------------------
      vector<vector<float>> refreshedInputMatrix;
      ovalCenterX += ovalDX;
      ovalCenterY += ovalDY;
      if ((((ovalCenterX - ovalRadiusX) <= (canvasSize * -0.5)) && (ovalDX < 0.0))
        || (((ovalCenterX + ovalRadiusX) >= (canvasSize * 0.5)) && (ovalDX > 0.0))) {
          ovalDX *= -1;
      }
      if ((((ovalCenterY - ovalRadiusY) <= (canvasSize * -0.5)) && (ovalDY < 0.0))
        || (((ovalCenterY + ovalRadiusY) >= (canvasSize * 0.5)) && (ovalDY > 0.0))) {
          ovalDY *= -1;
      }
      if (frameIndex > ovalNextChangeTimeX) {
        ovalSizeChangeDirectionX *= -1;
        ovalNextChangeTimeX = frameIndex + CHANGE_MOTION_LOWER_LIMIT + rnd::uniform(CHANGE_MOTION_UPPER_LIMIT - CHANGE_MOTION_LOWER_LIMIT);
      }
      if (frameIndex > ovalNextChangeTimeY) {
        ovalSizeChangeDirectionY *= -1;
        ovalNextChangeTimeY = frameIndex + CHANGE_MOTION_LOWER_LIMIT + rnd::uniform(CHANGE_MOTION_UPPER_LIMIT - CHANGE_MOTION_LOWER_LIMIT);
      }
      if (ovalRadiusX > canvasSize * OVAL_LARGEST) {
        ovalRadiusX = canvasSize * OVAL_LARGEST;
      }
      if (ovalRadiusX < canvasSize * OVAL_SMALLEST) {
        ovalRadiusX = canvasSize * OVAL_SMALLEST;
      }
      if (ovalRadiusY > canvasSize * OVAL_LARGEST) {
        ovalRadiusY = canvasSize * OVAL_LARGEST;
      }
      if (ovalRadiusY < canvasSize * OVAL_SMALLEST) {
        ovalRadiusY = canvasSize * OVAL_SMALLEST;
      }

      // Construct a 2D array: [LAYER][POSITIONS OF FIRED NEURONS IN THIS LAYER]
      vector<vector<Vec3f>> currentFiredNeuronPos;
      
      // If a point in the input layer is inside the oval
      // Set that value as [1.0], otherwise set it as [0.0]
      // And brush it as light yellow
      InputLayer.colors().clear();

      vector<Vec3f> currentFiredInputNeuronPos;
      
      for (int row = 0; row < ANN_SIZE; row++) {
        vector<float> refreshedOneLine;
        for (int col = 0; col < ANN_SIZE; col++) {
          float x = (row - (ANN_SIZE * 0.5)) * pointDistance;
          float y = (col - (ANN_SIZE * 0.5)) * pointDistance;
          float distanceX = abs(x - ovalCenterX) / ovalRadiusX;
          float distanceY = abs(y - ovalCenterY) / ovalRadiusY;
          float distance = sqrt(distanceX * distanceX + distanceY * distanceY);
          if (distance <= 1) {
            InputLayer.color(HSV(0.17f, 1.0f, 1.0f));
            refreshedOneLine.push_back(1.0);
            state().inputLayerNeuronRealTimeColor[row][col] = HSV(0.17f, 1.0f, 1.0f);
            currentFiredInputNeuronPos.push_back(state().inputLayerNeuronFixedPosition[row][col]);
          } else {
            InputLayer.color(HSV(0.0f, 0.0f, 0.7f));
            refreshedOneLine.push_back(0.0);
            state().inputLayerNeuronRealTimeColor[row][col] = HSV(0.0f, 0.0f, 0.7f);
          }
        }
        refreshedInputMatrix.push_back(refreshedOneLine);
      }

      currentFiredNeuronPos.push_back(currentFiredInputNeuronPos);
      liveInputMatrix = refreshedInputMatrix;


      // Step (3): feed the new input into the neural network -------------------------
      hiddenAndOutputNeurons.refreshInput(liveInputMatrix);
      

      // Step (4): refresh the list of firing neurons and the "lines" -----------------
      vector<vector<vector<float>>> currentNonInputLayerValues = hiddenAndOutputNeurons.getAllLayerOutput();

      OutputLayer.colors().clear();
      HiddenLayers.colors().clear();
      for (int layer = 0; layer <= ANN_NUM_HIDDEN_LAYERS; layer++) {
        vector<Vec3f> currentFiredNonInputNeuronPos;
        for (int row = 0; row < ANN_SIZE; row++) {
          for (int col = 0; col < ANN_SIZE; col++) {
            if (layer == ANN_NUM_HIDDEN_LAYERS) {
              if (currentNonInputLayerValues[ANN_NUM_HIDDEN_LAYERS][row][col] > outputLayerFireThreshold) {
                OutputLayer.color(HSV(0.17f, 1.0f, 1.0f));
                currentFiredNonInputNeuronPos.push_back(state().outputLayerNeuronFixedPosition[row][col]);

                int xDistToCenter = abs(col - (int)(0.5 * ANN_SIZE));
                int yDistToCenter = abs(row - (int)(0.5 * ANN_SIZE));
                int midiNote = 36 + 2 * (int)(sqrt(xDistToCenter * xDistToCenter + yDistToCenter * yDistToCenter));
                if (midiNote > 0)
                  {
                    synthManager.voice()->setInternalParameterValue(
                      "frequency", ::pow(2.f, (midiNote - 69.f) / 12.f) * 432.f);
                    synthManager.triggerOn(midiNote);
                   
                   MIDINoteTriggeredLastTime.push_back(midiNote);
                  }
                  
                
              } else {
                OutputLayer.color(HSV(0.0f, 0.0f, 0.7f));
              }
            } else {
              if (layer > 0) {
                if (currentNonInputLayerValues[layer][row][col] > firingThreshold) {
                  HiddenLayers.color(HSV(0.17f, 0.8f, 1.0f));
                  currentFiredNonInputNeuronPos.push_back(state().hiddenLayerNeuronFixedPosition[layer][row][col]);
                } else {
                  HiddenLayers.color(HSV(0.0f, 1.0f, 0.3f)); 
                }
              } else {
                if (currentNonInputLayerValues[layer][row][col] > inputLayerFireThreshold) {
                  HiddenLayers.color(HSV(0.17f, 0.8f, 1.0f));
                  currentFiredNonInputNeuronPos.push_back(state().hiddenLayerNeuronFixedPosition[layer][row][col]);
                } else {
                  HiddenLayers.color(HSV(0.0f, 1.0f, 0.3f)); 
                }
              }
              
            }  
          }
        }
        currentFiredNeuronPos.push_back(currentFiredNonInputNeuronPos);
      }

      // cout << "number of layers of fired neuron pos: " << currentFiredNeuronPos.size() << endl;
      // for (int i = 0; i < currentFiredNeuronPos.size(); i++) {
      //   int currentLayerNeuronsFired = currentFiredNeuronPos[i].size();
      //   cout << i << "'s layer fired " << currentLayerNeuronsFired << " neurons." << endl;
      // }


      //ConnectionLines.primitive(Mesh::LINES);
      ConnectionLines.colors().clear();
      ConnectionLines.vertices().clear();

      //for (int startLayer = 0; startLayer < currentFiredNeuronPos.size() - 1; startLayer++) {
      for (int startLayer = 0; startLayer < currentFiredNeuronPos.size() - 1; startLayer++) {
        
        vector<Vec3f> startLayerPositions = currentFiredNeuronPos[startLayer];
        vector<Vec3f> endLayerPositions = currentFiredNeuronPos[startLayer + 1];

        for (Vec3f oneStartPosition : startLayerPositions) {
          for (Vec3f oneEndPosition : endLayerPositions) {
            ConnectionLines.vertex(oneStartPosition);
            ConnectionLines.color(HSV(0.17f, 1.0f, 1.0f));
            ConnectionLines.vertex(oneEndPosition);
            ConnectionLines.color(HSV(0.17f, 1.0f, 1.0f));
          }
        }
      }
      ConnectionLines.primitive(Mesh::LINES);

    } else {
      frameCount += 1;
    }
    

    // Draw a window that contains the synth control panel
    synthManager.drawSynthControlPanel();
    imguiEndFrame();
    navControl().active(navi);
  }


  // --------------------------------------------------------
  // onDraw
  // The graphics callback function.
  void onDraw(Graphics &g) override
  {
    g.clear(0.0);
    // synthManager.render(g); <- This is commented out because we show ANN but not the notes
    g.meshColor();
    g.draw(ConnectionLines);

    g.shader(pointShader);
    g.blending(true);
    g.blendTrans();
    g.depthTesting(true);
    g.shader().uniform("pointSize", 0.12);
    g.draw(InputLayer);
    g.shader().uniform("pointSize", 0.05);
    g.draw(HiddenLayers);
    g.shader().uniform("pointSize", 0.12);
    g.draw(OutputLayer);
    

    // Draw Spectrum
    // Commented out for testing drawing the meshes of ANN only
    /*
    mSpectrogram.reset();
    mSpectrogram.primitive(Mesh::LINE_STRIP);
    if (showSpectro)
    {
      for (int i = 0; i < FFT_SIZE / 2; i++)
      {
        mSpectrogram.color(HSV(0.5 - spectrum[i] * 100));
        mSpectrogram.vertex(i, spectrum[i], 0.0);
      }
      g.meshColor(); // Use the color in the mesh
      g.pushMatrix();
      g.translate(-5.0, -3, 0);
      g.scale(100.0 / FFT_SIZE, 100, 1.0);
      g.draw(mSpectrogram);
      g.popMatrix();
    }
    */
    // GUI is drawn here
    if (showGUI)
    {
      imguiDraw();
      // ? how to show the "gui"
      // defined on line (): auto& gui = GUIdomain->newGUI();
      // with more adjustable parameters?
    }
    
  }


  // This gets called whenever a MIDI message is received on the port
  void onMIDIMessage(const MIDIMessage &m)
  {
    switch (m.type())
    {
    case MIDIByte::NOTE_ON:
    {
      int midiNote = m.noteNumber();
      if (midiNote > 0 && m.velocity() > 0.001)
      {
        synthManager.voice()->setInternalParameterValue(
            "frequency", ::pow(2.f, (midiNote - 69.f) / 12.f) * 432.f);
        synthManager.voice()->setInternalParameterValue(
            "attackTime", 0.1/m.velocity());
        synthManager.triggerOn(midiNote);
        printf("On Note %u, Vel %f \n", m.noteNumber(), m.velocity());
      }
      else
      {
        synthManager.triggerOff(midiNote);
        printf("Off Note %u, Vel %f \n", m.noteNumber(), m.velocity());
      }
      break;
    }
    case MIDIByte::NOTE_OFF:
    {
      int midiNote = m.noteNumber();
      printf("Note OFF %u, Vel %f", m.noteNumber(), m.velocity());
      synthManager.triggerOff(midiNote);
      break;
    }
    default:;
    }
  }

  // Whenever a key is pressed, this function is called
  bool onKeyDown(Keyboard const &k) override
  {
    if (ParameterGUI::usingKeyboard())
    { // Ignore keys if GUI is using
      // keyboard
      return true;
    }
    if (!navi)
    {
      if (k.shift())
      {
        // If shift pressed then keyboard sets preset
        int presetNumber = asciiToIndex(k.key());
        synthManager.recallPreset(presetNumber);
      }
      else
      {
        // Otherwise trigger note for polyphonic synth
        int midiNote = asciiToMIDI(k.key());
        if (midiNote > 0)
        {
          synthManager.voice()->setInternalParameterValue(
              "frequency", ::pow(2.f, (midiNote - 69.f) / 12.f) * 432.f);
          synthManager.triggerOn(midiNote);
        }
      }
    }
    switch (k.key())
    {
    case ']':
      showGUI = !showGUI;
      break;
    case '[':
      showSpectro = !showSpectro;
      break;
    case '=':
      navi = !navi;
      break;
    }
    return true;
  }

  // Whenever a key is released this function is called
  bool onKeyUp(Keyboard const &k) override
  {
    int midiNote = asciiToMIDI(k.key());
    if (midiNote > 0)
    {
      synthManager.triggerOff(midiNote);
    }
    return true;
  }

  void onExit() override { imguiShutdown(); }
};


// slurp
// To slurp from a file
//
string slurp(string fileName) {
  fstream file(fileName);
  string returnValue = "";
  while (file.good()) {
    string line;
    getline(file, line);
    returnValue += line + "\n";
  }
  return returnValue;
}

int main()
{
  MyApp app;

  if (al::Socket::hostName() == "ar01.1g") {
    AudioDevice device = AudioDevice("ECHO X5");
    app.configureAudio(device, 44100, 128, device.channelsOutMax(), 2);
  } else {
    app.configureAudio(48000., 512, 2, 2);
  }
  app.start();
  return 0;
}
