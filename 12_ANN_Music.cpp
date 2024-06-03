// MAT594P, Spring 2024
// Yuehao Gao
// Designed based on Myungin Lee(2022) Sine Envelope with Visuals

// Allosphere Artificial Neural Network Illustration with Sonification
// Inspired by a YouTube Video: 
// https://www.youtube.com/watch?v=Tsvxx-GGlTg&t=1s

// ----------------------------------------------------------------
// Press '=' to enable/disable navigation
// Press '[' or ']' to turn on & off GUI 
// ----------------------------------------------------------------


// How to make .synthSequence notes 
// # The '>' command adds an offset time to all events following
// > 50 
// # The '=' command adds another existing .synthSequence file to be played at the offset time.
// For example, the underlying command plays "note_02.synthSequence" file at 9 sec.
// = 9 note_02 1 


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
#include "al/math/al_Random.hpp"
#include "al/scene/al_PolySynth.hpp"
#include "al/scene/al_SynthSequencer.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"

#include "ANN.cpp"   // The ANN Class defined

// using namespace gam;
using namespace al;
using namespace std;
#define FFT_SIZE 4048
#define ANN_SIZE 20
#define ANN_NUM_HIDDEN_LAYERS 5

const float pointSize = 1.0;
const float pointDistance = 5.0;
const float layerDistance = 7.0 * pointDistance;
const float lineWidth = 0.5;

int frameCount = 0;
const int framePerChange = 30;


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

    // This is a quick way to create parameters for the voice. Trigger
    // parameters are meant to be set only when the voice starts, i.e. they
    // are expected to be constant within a voice instance. (You can actually
    // change them while you are prototyping, but their changes will only be
    // stored and aplied when a note is triggered.)

    createInternalTriggerParameter("amplitude", 0.3, 0.0, 1.0);
    createInternalTriggerParameter("frequency", 60, 20, 5000);
    createInternalTriggerParameter("attackTime", 1.0, 0.01, 3.0);
    createInternalTriggerParameter("releaseTime", 3.0, 0.1, 10.0);
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
  // "OSC error: out of buffer memory"
  // I tried and found that the maximum accumulative length allowed
  // is only 82, which is not enough

  // Neuron Points
  Vec3f inputLayerNeuronFixedPosition[ANN_SIZE][ANN_SIZE];
  HSV inputLayerNeuronRealTimeColor[ANN_SIZE][ANN_SIZE];
  Vec3f hiddenLayerNeuronFixedPosition[ANN_NUM_HIDDEN_LAYERS][ANN_SIZE][ANN_SIZE];
  HSV hiddenLayerNeuronRealTimeColor[ANN_NUM_HIDDEN_LAYERS][ANN_SIZE][ANN_SIZE];
  Vec3f outputLayerNeuronFixedPosition[ANN_SIZE][ANN_SIZE];
  HSV outputLayerNeuronRealTimeColor[ANN_SIZE][ANN_SIZE];

  // Lines
  Vec3f linesStartingFixedPosition[ANN_NUM_HIDDEN_LAYERS + 1][ANN_SIZE][ANN_SIZE];
  Vec3f linesEndingFixedPosition[ANN_NUM_HIDDEN_LAYERS + 1][ANN_SIZE][ANN_SIZE];
  HSV linesRealTimeColor[ANN_NUM_HIDDEN_LAYERS + 1][ANN_SIZE][ANN_SIZE];
  
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
  // GUI manager for SineEnv voices
  SynthGUIManager<SineEnv> synthManager{"SineEnv"};

  // The hidden neurons that producing the data output
  NonInputLayers hiddenLayersNeurons = NonInputLayers(ANN_NUM_HIDDEN_LAYERS, ANN_SIZE);

  RtMidiIn midiIn; // MIDI input carrier
  Mesh mSpectrogram;
  vector<float> spectrum;
  bool showGUI = true;
  bool showSpectro = true;
  bool navi = false;

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

    // ------------------------------------------------------------
    // Initialize parameters for all meshes
    
    
    
    // The input layer
    for (int ipRow = 0; ipRow < ANN_SIZE; ipRow++) {
      for (int ipColumn = 0; ipColumn < ANN_SIZE; ipColumn++) {
        float x = (ipRow - (ANN_SIZE * 0.5)) * pointDistance;
        float y = (ipColumn - (ANN_SIZE * 0.5)) * pointDistance;
        float z = 0.0;

        InputLayer.vertex(Vec3f(x, y, z));
        state().inputLayerNeuronFixedPosition[ipRow][ipColumn] = Vec3f(x, y, z);
        InputLayer.color(HSV(60.0, 100.0, 100.0));  // Input layer initialized as white
        state().inputLayerNeuronRealTimeColor[ipRow][ipColumn] = HSV(60.0f, 100.0f, 100.0f);
        InputLayer.texCoord(1.0, 0);
      }
    }

    // The hidden layers
    for (int hdLayer = 0; hdLayer < ANN_NUM_HIDDEN_LAYERS; hdLayer++) {
      for (int hdRow = 0; hdRow < ANN_SIZE; hdRow++) {
        for (int hdColumn = 0; hdColumn < ANN_SIZE; hdColumn++) {
          float x = (hdRow - (ANN_SIZE * 0.5)) * pointDistance;
          float y = (hdColumn - (ANN_SIZE * 0.5)) * pointDistance;
          float z = hdLayer * layerDistance;

          HiddenLayers.vertex(Vec3f(x, y, z));
          state().hiddenLayerNeuronFixedPosition[hdLayer][hdRow][hdColumn] = Vec3f(x, y, z);
        
          HiddenLayers.color(HSV(0, 0, 100));  // Input layer initialized as white
          state().hiddenLayerNeuronRealTimeColor[hdLayer][hdRow][hdColumn] = HSV(0.0f, 0.0f, 100.0f);
        }
      }
    }

    // The output layer
    for (int opRow = 0; opRow < ANN_SIZE; opRow++) {
      for (int opColumn = 0; opColumn < ANN_SIZE; opColumn++) {
        float x = (opRow - (ANN_SIZE * 0.5)) * pointDistance;
        float y = (opColumn - (ANN_SIZE * 0.5)) * pointDistance;
        float z = (ANN_NUM_HIDDEN_LAYERS + 1) * layerDistance;

        OutputLayer.vertex(Vec3f(x, y, z));
        state().outputLayerNeuronFixedPosition[opRow][opColumn] = Vec3f(x, y, z);
        
        OutputLayer.color(HSV(0, 0, 100));  // Input layer initialized as white
        state().inputLayerNeuronRealTimeColor[opRow][opColumn] = HSV(0, 0, 100);
      }
    }

    

    InputLayer.primitive(Mesh::POINTS);
    HiddenLayers.primitive(Mesh::POINTS);
    OutputLayer.primitive(Mesh::POINTS);
    ConnectionLines.primitive(Mesh::LINES);
    
    
    // ------------------------------------------------------------

    navControl().active(false); // Disable navigation via keyboard, since we
                                // will be using keyboard for note triggering
    // Set sampling rate for Gamma objects from app's audio
    gam::sampleRate(audioIO().framesPerSecond());
    imguiInit();
    // Play example sequence. Comment this line to start from scratch
    synthManager.synthRecorder().verbose(true);

    if (isPrimary()) {
      nav().pos(0.0, 0.0, 12.0);
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

        //?
        // Set up GUI
        // auto GUIdomain = GUIDomain::enableGUI(defaultWindowDomain());
        // auto& gui = GUIdomain->newGUI();


        // Open the last device found
        unsigned int port = midiIn.getPortCount() - 1;
        midiIn.openPort(port);
        printf("Opened port to %s\n", midiIn.getPortName(port).c_str());
      }
      else
      {
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
    g.shader(pointShader);
    g.shader().uniform("pointSize", 1.0);
    g.blending(true);
    g.blendTrans();
    g.depthTesting(true);
    g.draw(InputLayer);

    // Draw Spectrum
    // Commented out for testing drawing the meshes of ANN only
    ///*
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
    // GUI is drawn here
    if (showGUI)
    {
      imguiDraw();
      // ? how to show the "gui"
      // defined on line (): auto& gui = GUIdomain->newGUI();
      // with more adjustable parameters?
    }
    //*/
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
  // Create app instance
  // MyApp app;

  // // Set up audio
  // app.configureAudio(48000., 512, 2, 0);
  // app.start();
  // return 0;

  MyApp app;

  //NonInputLayers hiddenLayersNeurons(5, 20);


  if (al::Socket::hostName() == "ar01.1g") {
    AudioDevice device = AudioDevice("ECHO X5");
    app.configureAudio(device, 44100, 128, device.channelsOutMax(), 2);
  } else {
    app.configureAudio(48000., 512, 2, 2);
  }
  app.start();
  return 0;
}
