
/***
## Declare Factory
 
Create the Factory class. Later you can choose the methods
whose performance you'd like to investigate.
 
The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass
 
 - The first argument is the base of the name of all the output
weightfiles in the directory weight/ that will be created with the
method parameters
 
 - The second argument is the output file for the training results
 
 - The third argument is a string option defining some general configuration for the TMVA session. For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the option string
 
**/

/**
void tree3AddBranch(FileName) {
    TFile f(FileName, "update");
 
    Float_t new_v;
    auto t3 = f->Get<TTree>("DiphotonTree/data_125_13TeV_NOTAG;1");
    auto newBranch = t3->Branch("dijet_ptOverMjjgg", &dijet_ptOverMjjgg, "dijet_ptOverMjjgg/F");
 
    Long64_t nentries = t3->GetEntries(); // read the number of entries in the t3
    double_t dijet_mass, mass, HHbbggCandidate_mass

    data_tree->SetBranchAddress("dijet_mass", &dijet_mass);
    data_tree->SetBranchAddress("mass", &mass);
    data_tree->SetBranchAddress("HHbbggCandidate_mass", &HHbbggCandidate_mass);


    for (Long64_t i = 0; i < nentries; i++) {
        dijet_ptOverMjjgg = ;
        newBranch->Fill();
    }
 
    t3->Write("", TObject::kOverwrite); // save only the new version of the tree
}
**/

void TMVAClassificationNew() {
 
  // options to control used methods
 
  bool useLikelihood = false;    // likelihood based discriminant
  bool useLikelihoodKDE = false;    // likelihood based discriminant
  bool useFischer = false;       // Fischer discriminant
  bool useMLP = false;          // Multi Layer Perceptron (old TMVA NN implementation)
  bool useBDT = true;           // Boosted Decision Tree
  bool useDL = false;            // TMVA Deep learning ( CPU or GPU)
  bool useKeras = false;        // Keras Deep learning
  bool usePyTorch = false;      // PyTorch Deep learning
 
  TMVA::Tools::Instance();

#ifdef R__HAS_PYMVA
  gSystem->Setenv("KERAS_BACKEND", "tensorflow");
  // for using Keras
  TMVA::PyMethodBase::PyInitialize();
#else
  useKeras = false;
  usePyTorch = false;
#endif

  auto outputFile = TFile::Open("HH_ClassificationOutput.root", "RECREATE");

 
  TMVA::Factory factory("TMVA_Higgs_Classification", outputFile,
			"!V:ROC:!Silent:Color:AnalysisType=Classification" );
 
  /**
 
## Setup Dataset(s)
 
Define now input data file and signal and background trees
 
  **/
  /*
  TString inputFileName = "Higgs_data.root";
  TString inputFileLink = "http://root.cern/files/" + inputFileName;
 
  TFile *inputFile = nullptr;
 
  if (!gSystem->AccessPathName(inputFileName)) {
    // file exists
    inputFile = TFile::Open( inputFileName );
  }
 
  if (!inputFile) {
    // download file from Cernbox location
    Info("TMVA_Higgs_Classification","Download Higgs_data.root file");
    TFile::SetCacheFileDir(".");
    inputFile = TFile::Open(inputFileLink, "CACHEREAD");
    if (!inputFile) {
      Error("TMVA_Higgs_Classification","Input file cannot be downloaded - exit");
      return;
    }
  }
  */
   //TString fnameS = "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v1/Run3_2022postEE/GluGluToHH/nominal/31f7eda6-ad77-11ee-9927-57e36880beef_%2FEvents%3B1_0-210.parquet";
   TString fnameS = "/eos/user/e/ejourdhu/HHbbgg_root/GluGluHToHH.root";

   //TString fnameB = "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v1/Run3_2022postEE/GGJets/nominal/3858888e-63c8-11ee-91ba-b72411acbeef_%2FEvents%3B1_0-4928.parquet";
   //TString fnameB = "/eos/user/e/ejourdhu/HHbbgg_root/GGJets.root";
   //TString fnameB = "/eos/user/e/ejourdhu/HHbbgg_root/VHToGG_m120.root";
   TString fnameB = "/eos/user/e/ejourdhu/HHbbgg_root/VBFHToGG.root";


  TFile *inputS = TFile::Open( fnameS );
   TFile *inputB = TFile::Open( fnameB );
   
   std::cout << "--- TMVAClassification       : Using signal input file: " << inputS->GetName() << std::endl;
   std::cout << "--- TMVAClassification       : Using background input file: " << inputB->GetName() << std::endl;
   
   // --- Register the training and test trees

   TTree *signalTree    = (TTree*)inputS->Get("DiphotonTree/data_125_13TeV_NOTAG;1");
   TTree *backgroundTree= (TTree*)inputB->Get("DiphotonTree/data_125_13TeV_NOTAG;1");

  // --- Register the training and test trees
 
   //  TTree *signalTree     = (TTree*)inputFile->Get("sig_tree");
   //TTree *backgroundTree = (TTree*)inputFile->Get("bkg_tree");
 
  signalTree->Print();
 
  /***
## Declare DataLoader(s)
 
The next step is to declare the DataLoader class that deals with input variables
 
Define the input variables that shall be used for the MVA training
note that you may also use variable expressions, which can be parsed by TTree::Draw( "expression" )]
 
  ***/
 
  TMVA::DataLoader * loader = new TMVA::DataLoader("dataset");


 
  loader->AddVariable("lead_bjet_pt");
  loader->AddVariable("lead_bjet_phi");
  loader->AddVariable("lead_bjet_eta");
  loader->AddVariable("lead_bjet_btagPNetB");
  loader->AddVariable("sublead_bjet_pt");
  loader->AddVariable("sublead_bjet_phi");
  loader->AddVariable("sublead_bjet_eta");
  loader->AddVariable("sublead_bjet_btagPNetB");
  //loader->AddVariable("CosThetaStar_CS");
  //loader->AddVariable("CosThetaStar_gg");
  //loader->AddVariable("CosThetaStar_jj");
  loader->AddVariable("DeltaR_jg_min");
  loader->AddVariable("lead_pt");
  loader->AddVariable("lead_phi");
  loader->AddVariable("lead_eta");
  loader->AddVariable("sublead_pt");
  loader->AddVariable("sublead_phi");
  loader->AddVariable("sublead_eta");

  //loader->AddVariable("pholead_PtOverM");
  //loader->AddVariable("phosublead_PtOverM");
  //loader->AddVariable("FirstJet_PtOverM");
  //loader->AddVariable("SecondJet_PtOverM");
  //loader->AddVariable("fixedGridRhoAll");


 
  /// We set now the input data trees in the TMVA DataLoader class
 
  // global event weights per tree (see below for setting event-wise weights)
  Double_t signalWeight     = 1.0;
  Double_t backgroundWeight = 1.0;
 
  // You can add an arbitrary number of signal or background trees
  loader->AddSignalTree    ( signalTree,     signalWeight     );
  loader->AddBackgroundTree( backgroundTree, backgroundWeight );
 
 
  // Set individual event weights (the variables must exist in the original TTree)
  //    for signal    : factory->SetSignalWeightExpression    ("weight1*weight2");
  //    for background: factory->SetBackgroundWeightExpression("weight1*weight2");
  //loader->SetBackgroundWeightExpression( "weight" );
 
  // Apply additional cuts on the signal and background samples (can be different)
  TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
  TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";
 
  // Tell the factory how to use the training and testing events
  //
  // If no numbers of events are given, half of the events in the tree are used
  // for training, and the other half for testing:
  //    loader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
  // To also specify the number of testing events, use:
 
  loader->PrepareTrainingAndTestTree( mycuts, mycutb,
				      "nTrain_Signal=20000:nTrain_Background=20000:nTest_Signal=20000:nTest_Background=20000:SplitMode=Random:!V" );
 
  /***
## Booking Methods
 
Here we book the TMVA methods. We book first a Likelihood based on KDE (Kernel Density Estimation), a Fischer discriminant, a BDT
and a shallow neural network
 
  */
 
 
  // Likelihood ("naive Bayes estimator")
  if (useLikelihood) {
    factory.BookMethod(loader, TMVA::Types::kLikelihood, "Likelihood",
		       "H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );
  }
  // Use a kernel density estimator to approximate the PDFs
  if (useLikelihoodKDE) {
    factory.BookMethod(loader, TMVA::Types::kLikelihood, "LikelihoodKDE",
		       "!H:!V:!TransformOutput:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Adaptive:KDEFineFactor=0.3:KDEborder=None:NAvEvtPerBin=50" );
 
  }
 
  // Fisher discriminant (same as LD)
  if (useFischer) {
    factory.BookMethod(loader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );
  }
 
  //Boosted Decision Trees
  if (useBDT) {
    factory.BookMethod(loader,TMVA::Types::kBDT, "BDT",
		       "!V:NTrees=700:MinNodeSize=3%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );
  }
 
  //Multi-Layer Perceptron (Neural Network)
  if (useMLP) {
    factory.BookMethod(loader, TMVA::Types::kMLP, "MLP",
		       "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
  }
 
 
  /// Here we book the new DNN of TMVA if we have support in ROOT. We will use GPU version if ROOT is enabled with GPU
 
 
  /***
 
## Booking Deep Neural Network
 
Here we define the option string for building the Deep Neural network model.
 
#### 1. Define DNN layout
 
The DNN configuration is defined using a string. Note that whitespaces between characters are not allowed.
 
We define first the DNN layout:
 
- **input layout** :   this defines the input data format for the DNN as  ``input depth | height | width``.
   In case of a dense layer as first layer the input layout should be  ``1 | 1 | number of input variables`` (features)
- **batch layout**  : this defines how are the input batch. It is related to input layout but not the same.
   If the first layer is dense it should be ``1 | batch size ! number of variables`` (features)
 
   *(note the use of the character `|` as  separator of  input parameters for DNN layout)*
 
note that in case of only dense layer the input layout could be omitted but it is required when defining more
complex architectures
 
- **layer layout** string defining the layer architecture. The syntax is
   - layer type (e.g. DENSE, CONV, RNN)
   - layer parameters (e.g. number of units)
   - activation function (e.g  TANH, RELU,...)
 
   *the different layers are separated by the ``","`` *
 
#### 2. Define Training Strategy
 
We define here the training strategy parameters for the DNN. The parameters are separated by the ``","`` separator.
One can then concatenate different training strategy with different parameters. The training strategy are separated by
the ``"|"`` separator.
 
 - Optimizer
 - Learning rate
 - Momentum (valid for SGD and RMSPROP)
 - Regularization and Weight Decay
 - Dropout
 - Max number of epochs
 - Convergence steps. if the test error will not decrease after that value the training will stop
 - Batch size (This value must be the same specified in the input layout)
 - Test Repetitions (the interval when the test error will be computed)
 
 
#### 3. Define general DNN options
 
We define the general DNN options concatenating in the final string the previously defined layout and training strategy.
Note we use the ``":"`` separator to separate the different higher level options, as in the other TMVA methods.
In addition to input layout, batch layout and training strategy we add now:
 
- Type of Loss function (e.g. CROSSENTROPY)
- Weight Initizalization (e.g XAVIER, XAVIERUNIFORM, NORMAL )
- Variable Transformation
- Type of Architecture (e.g. CPU, GPU, Standard)
 
We can then book the DL method using the built option string
 
  ***/
 
  if (useDL) {
 
    bool useDLGPU = false;
#ifdef R__HAS_TMVAGPU
    useDLGPU = true;
#endif
 
    // Define DNN layout
    TString inputLayoutString = "InputLayout=1|1|7";
    TString batchLayoutString= "BatchLayout=1|128|7";
    TString layoutString ("Layout=DENSE|64|TANH,DENSE|64|TANH,DENSE|64|TANH,DENSE|64|TANH,DENSE|1|LINEAR");
    // Define Training strategies
    // one can catenate several training strategies
    TString training1("LearningRate=1e-3,Momentum=0.9,"
                        "ConvergenceSteps=10,BatchSize=128,TestRepetitions=1,"
                        "MaxEpochs=30,WeightDecay=1e-4,Regularization=None,"
		      "Optimizer=ADAM,ADAM_beta1=0.9,ADAM_beta2=0.999,ADAM_eps=1.E-7," // ADAM default parameters
		      "DropConfig=0.0+0.0+0.0+0.");
    //     TString training2("LearningRate=1e-3,Momentum=0.9"
    //                       "ConvergenceSteps=10,BatchSize=128,TestRepetitions=1,"
    //                       "MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"
    //                       "Optimizer=SGD,DropConfig=0.0+0.0+0.0+0.");
 
    TString trainingStrategyString ("TrainingStrategy=");
    trainingStrategyString += training1; // + "|" + training2;
 
    // General Options.
 
    TString dnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=G:"
			"WeightInitialization=XAVIER");
    dnnOptions.Append (":"); dnnOptions.Append (inputLayoutString);
    dnnOptions.Append (":"); dnnOptions.Append (batchLayoutString);
    dnnOptions.Append (":"); dnnOptions.Append (layoutString);
    dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString);
 
    TString dnnMethodName = "DNN_CPU";
    if (useDLGPU) {
      dnnOptions += ":Architecture=GPU";
      dnnMethodName = "DNN_GPU";
    } else  {
      dnnOptions += ":Architecture=CPU";
    }
 
    //    factory.BookMethod(loader, TMVA::Types::kDL, dnnMethodName, dnnOptions);
  }
 
  // Keras deep learning
  if (useKeras) {
 
    Info("TMVA_Higgs_Classification", "Building deep neural network with keras ");
    // create python script which can be executed
    // create 2 conv2d layer + maxpool + dense
    TMacro m;
    m.AddLine("import tensorflow");
    m.AddLine("from tensorflow.keras.models import Sequential");
    m.AddLine("from tensorflow.keras.optimizers import Adam");
    m.AddLine("from tensorflow.keras.layers import Input, Dense");
    m.AddLine("");
    m.AddLine("model = Sequential() ");
    m.AddLine("model.add(Dense(64, activation='relu',input_dim=7))");
    m.AddLine("model.add(Dense(64, activation='relu'))");
    m.AddLine("model.add(Dense(64, activation='relu'))");
    m.AddLine("model.add(Dense(64, activation='relu'))");
    m.AddLine("model.add(Dense(2, activation='sigmoid'))");
    m.AddLine("model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.001), weighted_metrics = ['accuracy'])");
    m.AddLine("model.save('Higgs_model.h5')");
    m.AddLine("model.summary()");
 
    m.SaveSource("make_higgs_model.py");
    // execute
    auto ret = (TString *)gROOT->ProcessLine("TMVA::Python_Executable()");
    TString python_exe = (ret) ? *(ret) : "python";
    gSystem->Exec(python_exe + " make_higgs_model.py");
 
    if (gSystem->AccessPathName("Higgs_model.h5")) {
      Warning("TMVA_Higgs_Classification", "Error creating Keras model file - skip using Keras");
    } else {
      // book PyKeras method only if Keras model could be created
      Info("TMVA_Higgs_Classification", "Booking tf.Keras Dense model");
      factory.BookMethod(
			 loader, TMVA::Types::kPyKeras, "PyKeras",
            "H:!V:VarTransform=None:FilenameModel=Higgs_model.h5:tf.keras:"
            "FilenameTrainedModel=Higgs_trained_model.h5:NumEpochs=20:BatchSize=100:"
			 "GpuOptions=allow_growth=True"); // needed for RTX NVidia card and to avoid TF allocates all GPU memory
    }
  }
 
  /**
## Train Methods
 
Here we train all the previously booked methods.
 
  */
 
  factory.TrainAllMethods();
 
  /**
   ## Test  all methods
 
 Now we test and evaluate all methods using the test data set
  */
 
  factory.TestAllMethods();
 
  factory.EvaluateAllMethods();
 
  /// after we get the ROC curve and we display
 
  auto c1 = factory.GetROCCurve(loader);
  c1->Draw();
 
  /// at the end we close the output file which contains the evaluation result of all methods and it can be used by TMVAGUI
  /// to display additional plots
 
  outputFile->Close();
 
 
}
