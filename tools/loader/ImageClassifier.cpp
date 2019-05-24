/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Loader.h"

#include "glow/Base/Image.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

using namespace glow;

namespace {

/// Image loader options.
llvm::cl::OptionCategory imageLoaderCat("Image Loader Options");
llvm::cl::list<std::string> inputImageFilenames(
    llvm::cl::Positional,
    llvm::cl::desc("<input files> (note: specifying '-' enables streaming "
                   "mode, where the model is compiled once and then can be run "
                   "many times with new input filenames passed via stdin)"),
    llvm::cl::ZeroOrMore);

llvm::cl::opt<std::string> inputImageListFile(
    "input-image-list-file",
    llvm::cl::desc(
        "Name of the file containing list of images (one image per line)"),
    llvm::cl::value_desc("string_name"), llvm::cl::Optional,
    llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> miniBatch(
    "minibatch",
    llvm::cl::desc(
        "Size of mini-batches. Split the input image list into a set of "
        "mini-batches. The input model is compiled for an input tensor batch "
        "size equal to the specified mini-batch size and mini-batches of "
        "images are inferred separatly. The number of input images must be a "
        "multiple of the mini-batch size. By default, splitting the input "
        "image list into mini-batches is deactivated."),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> miniBatchThreads(
    "minibatch-threads",
    llvm::cl::desc(
        "Max number of threads used to process mini-batches. If "
        "minibatch-threads is greater than 1, and we are working in minibatch "
        "mode, then several worker threads are created to process the "
        "minibatches. Then the minibatches are distributed between these "
        "threads, and each thread processes its set of minibatches "
        "independently."
        " By default, the number of threads is 1, and no parallelization is "
        "happening. These are things to be aware of:\n"
        "\t- The actual number of worker threads can be less than specified by "
        "this option (for example, if specified number of threads is greater "
        "than number of minibatches to process). Their number may also be "
        "forced to 1 in some cases (see below);\n"
        "\t- Currently, dumping profile and emitting bundle force "
        "single-threaded mode;\n"
        "\t- If a model has operations that make reduction across images in "
        "the batch, it's user's responsibility to make sure that this model is "
        "not processed in multi-threaded mode. Otherwise, the correctness of "
        "results is not guaranteed."),
    llvm::cl::Optional, llvm::cl::init(1), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned> labelOffset(
    "label-offset",
    llvm::cl::desc("Label offset for TF ONNX models with 1001 classes"),
    llvm::cl::Optional, llvm::cl::init(0), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<bool> computeSoftmax(
    "compute-softmax", llvm::cl::desc("Compute softmax of the network output"),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<unsigned>
    topKCount("topk",
              llvm::cl::desc("Number of highest likelihood labels to print and "
                             "match the correspondent expected-lables"),
              llvm::cl::Optional, llvm::cl::init(1),
              llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<std::string> modelInputName(
    "model-input-name",
    llvm::cl::desc("The name of the variable for the model's input image."),
    llvm::cl::value_desc("string_name"), llvm::cl::Required,
    llvm::cl::cat(imageLoaderCat));

llvm::cl::opt<bool> convertInAndOutToFp16(
    "convert-inout-to-fp16",
    llvm::cl::desc(
        "Convert the input and output tensors of the network to fp16"),
    llvm::cl::cat(imageLoaderCat));

llvm::cl::list<unsigned> expectedMatchingLabels(
    "expected-labels",
    llvm::cl::desc("The comma delimited list of the matching lables"),
    llvm::cl::value_desc("int"), llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(imageLoaderCat));
} // namespace

/// Write a prompt to stdout asking for filenames for classification. Read in
/// those filenames and add them to \p filenames. \p filenames is cleared before
/// adding the new set of filenames from stdin. \returns false if the passed in
/// line was empty.
static bool getNextImageFilenames(std::vector<std::string> *filenames) {
  // Clear out old filenames before adding new ones.
  filenames->clear();

  llvm::outs() << "Enter image filenames to classify: ";

  // Add in each filename to the vector.
  std::string filenamesRaw;
  getline(std::cin, filenamesRaw);
  std::istringstream iss(filenamesRaw);
  std::string filename;
  while (iss >> filename) {
    filenames->push_back(filename);
  }

  return !filenames->empty();
}

/// Generate in \p imageList the list of filenames corresponding to the next
/// mini-batch of size \p miniBatchSize extracted from \p totalImageList at

/// index \p minibatchIndex. /returns true if the index is valid, false
/// otherwise. In case the function returns true, \p minibatchIndex is
/// incremented by \p miniBatchSize. Stop when upon reaching \p miniBatchLimit.
static bool getNextMiniBatch(std::vector<std::string> &imageList,
                             std::vector<std::string> &totalImageList,
                             size_t &miniBatchIndex, size_t miniBatchSize,
                             size_t miniBatchLimit) {
  if (miniBatchIndex >= miniBatchLimit) {
    return false;
  }
  imageList.clear();
  size_t endIndex = miniBatchIndex + miniBatchSize;
  for (size_t index = miniBatchIndex; index < endIndex; index++) {
    imageList.push_back(totalImageList[index]);
  }
  miniBatchIndex += miniBatchSize;
  return true;
}

/// Creates and \returns the ProtobufLoader given \p loader and the
/// \p inputImageType. Note that this must come after loading images for
/// inference so that \p inputImageType is known.
static std::unique_ptr<ProtobufLoader>
createProtobufLoader(Loader &loader, TypeRef inputImageType) {
  // The image name that the model expects must be passed on the command line.
  const char *inputName = modelInputName.c_str();

  // Create the model based on the input model format.
  std::unique_ptr<ProtobufLoader> LD;
  bool c2Model = !loader.getCaffe2NetDescFilename().empty();
  if (c2Model) {
    LD.reset(new Caffe2ModelLoader(
        loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
        {inputName}, {inputImageType}, *loader.getFunction()));
  } else {
    LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), {inputName},
                                 {inputImageType}, *loader.getFunction()));
  }

  return LD;
}

/// Given \p loader, the \p bindings, and \p inputImageType, build the graph
/// from the provided protobuf file found via \p loader. Then compiles and
/// \returns a pair of pointers to the input Placeholder and output Tensor for
/// the Softmax.
static std::pair<Placeholder *, Tensor *>
buildAndCompileAndGetInAndOutPair(Loader &loader, PlaceholderBindings &bindings,
                                  TypeRef inputImageType) {
  auto LD = createProtobufLoader(loader, inputImageType);

  // Allocate tensors to back all inputs and outputs.
  bindings.allocate(loader.getModule()->getPlaceholders());

  // Convert the placeholders for now. The backing Tensor's data will be
  // converted later.
  if (convertInAndOutToFp16) {
    TypeAToTypeBFunctionConverter converter(
        *loader.getFunction(), ElemKind::FloatTy, ElemKind::Float16Ty);
    for (auto *placeholder : loader.getModule()->getPlaceholders()) {
      converter.convertPlaceholder(*placeholder, &bindings);
    }
  }

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  loader.compile(bindings);

  // The image name that the model expects must be passed on the command line.
  const char *inputName = modelInputName.c_str();
  Placeholder *inputImagePH =
      llvm::cast<Placeholder>(EXIT_ON_ERR(LD->getNodeValueByName(inputName)));

  // Get the Tensor from the Placeholder that the final expected Softmax writes
  // into at the end of image inference.
  Placeholder *SMPH = EXIT_ON_ERR(LD->getSingleOutput());
  Tensor *SMT = bindings.get(SMPH);

  return std::make_pair(inputImagePH, SMT);
}

/// A pair representing a float and the index where the float was found.
using FloatIndexPair = std::pair<float, size_t>;

/// Given a Handle \p H of a 1D tensor with float elements, \returns the top K
/// (topKCount) [float, index] pairs, i.e. the pairs with the highest floats.
template <typename ElemTy>
static std::vector<FloatIndexPair> getTopKPairs(Handle<ElemTy> H) {
  assert(topKCount <= H.size() && "Function requires k < number of labels.");
  assert(H.dims().size() == 1 && "H must be a Handle of a 1d Tensor.");

  // Use a priority queue of pairs of floats (probabilities) to size_t (indices)
  // to determine the top K pairs, and then return the indices from it.
  std::priority_queue<FloatIndexPair, std::vector<FloatIndexPair>,
                      std::greater<FloatIndexPair>>
      topKQueue;

  // Loop over all the probabilites, finding the highest k probability pairs.
  for (size_t i = 0, e = H.size(); i < e; i++) {
    float currProbability = H.at({i});
    if (topKQueue.size() < topKCount) {
      // Always push the first k elements.
      topKQueue.push(std::make_pair(currProbability, i));
    } else if (topKQueue.top().first < currProbability) {
      // If the lowest element has lower probability than the current, then pop
      // the lowest and insert the current pair.
      topKQueue.pop();
      topKQueue.push(std::make_pair(currProbability, i));
    }
  }

  // We now have the top K pairs in reverse order.
  std::vector<FloatIndexPair> res(topKCount);
  for (size_t i = 0; i < topKCount; i++) {
    res[topKCount - i - 1] = topKQueue.top();
    topKQueue.pop();
  }

  return res;
}

/// Print out the top K pairs to stdout, which were passed in via \p topKPairs.
static void printTopKPairs(const std::vector<FloatIndexPair> &topKPairs) {
  for (size_t i = 0; i < topKPairs.size(); i++) {
    // Some models are trained with more classes. E.g. Some imagenet models
    // exported from TensorFlow have 1 extra "neutral" class.
    const size_t label = topKPairs[i].second - labelOffset;
    // Tab out the label so it aligns nicely with Label-K1.
    if (i != 0) {
      llvm::outs() << "\t\t\t\t\t";
    }
    llvm::outs() << "\tLabel-K" << i + 1 << ": " << label << " (probability: "
                 << llvm::format("%0.4f", topKPairs[i].first) << ")\n";
  }
}

/// Checks if \p topKPairs have the index that matches the provided index,
/// \returns 0 on success and 1 if mismatches found.
static int checkExpectedLabel(llvm::ArrayRef<FloatIndexPair> topKPairs,
                              llvm::StringRef fileName,
                              unsigned expectedCategoryIndex) {
  // Loop through pairs and try to find a matching label.
  for (const auto &p : topKPairs) {
    if (p.second - labelOffset == expectedCategoryIndex) {
      return 0;
    }
  }

  llvm::outs() << " File: " << fileName
               << " doesn't match index: " << expectedCategoryIndex
               << " in the top " << topKPairs.size() << " pairs\n";

  return 1;
}

/// Apply the softmax function to the given handle.
template <typename ElemTy> static void applySoftmax(Handle<ElemTy> H) {
  assert(H.dims().size() == 1 && "H must be a Handle of a 1d Tensor.");
  float denominator = 0.0f;

  for (auto elem : H) {
    denominator += std::exp(static_cast<float>(elem));
  }

  for (auto &elem : H) {
    elem = std::exp(static_cast<float>(elem)) / denominator;
  }
}

/// Given the output Softmax Tensor \p SMT and \p imageList, prints the
/// results of inference and returns number of incorrect predictions,
/// \returns the number of found mismatches.
template <typename ElemTy>
static int processAndPrintResultsImpl(Tensor *SMT,
                                      llvm::ArrayRef<std::string> imageList) {
  // Softmax should have at least two dims: batchSize, numLabels, and then
  // optionally trailing 1s.
  assert(SMT->dims().size() >= 2 && "Softmax should have at least 2 dims.");
  const size_t batchSize = SMT->dims()[0];
  (void)batchSize;
  assert(batchSize == imageList.size() &&
         "Softmax batch size must equal the input number of images.");
  for (size_t i = 2; i < SMT->dims().size(); i++) {
    assert(SMT->dims()[i] == 1 && "Trailing dims must be 1 for Softmax.");
  }
  const size_t numLabels = SMT->dims()[1];
  std::vector<size_t> sliceOffset(SMT->dims().size(), 0);

  int retVal = 0;
  for (unsigned i = 0; i < imageList.size(); i++) {
    const auto &fileName = imageList[i];
    llvm::outs() << " File: " << fileName;

    // batchSize is the first dimension, so update it to get the next slice.
    sliceOffset[0] = i;
    Tensor slice = SMT->getUnowned({numLabels}, sliceOffset);
    auto SH = slice.getHandle<ElemTy>();

    if (computeSoftmax) {
      applySoftmax(SH);
    }

    auto topKPairs = getTopKPairs(SH);
    printTopKPairs(topKPairs);
    if (!expectedMatchingLabels.empty()) {
      retVal +=
          checkExpectedLabel(topKPairs, fileName, expectedMatchingLabels[i]);
    }
  }

  return retVal;
}

/// Given the output Softmax Tensor \p SMT and \p functionName, switch between
/// the correct element type to print the results of inference as contained in
/// \p SMT, \returns the number of found mismatches.
static int processAndPrintResults(Tensor *SMT,
                                  llvm::ArrayRef<std::string> imageList) {
  switch (SMT->getElementType()) {
  case ElemKind::FloatTy:
    return processAndPrintResultsImpl<float>(SMT, imageList);
  case ElemKind::Float16Ty:
    return processAndPrintResultsImpl<float16_t>(SMT, imageList);
  default:
    llvm_unreachable("Type not supported");
  }
}

/// Read all images from \p inputImageListFile in to \p inputImageFilenames.
static void parseInputImageList(const std::string &inputImageListFile) {
  std::ifstream inFile;
  inFile.open(inputImageListFile);
  while (!inFile.eof()) {
    std::string img;
    getline(inFile, img);
    if (!img.empty()) {
      inputImageFilenames.push_back(img);
    }
  }
  inFile.close();
}

struct ScopedLock {
  std::mutex &mu;
  ScopedLock(std::mutex &mu) : mu(mu) { mu.lock(); }
  ~ScopedLock() { mu.unlock(); }
};

int main(int argc, char **argv) {
  // Verify/initialize command line parameters, and then loader initializes
  // the ExecutionEngine and Function.
  parseCommandLine(argc, argv);

  if (inputImageListFile.empty() && inputImageFilenames.size() == 0) {
    llvm::errs()
        << "Args: Either positional inputImageFilenames or -inputImageListFile "
           "must be used to specify input images.\n";
    std::exit(1);
  }

  if (!inputImageListFile.empty()) {
    GLOW_ASSERT(
        inputImageFilenames.size() == 0 &&
        "When using -input-image-list-file all Input images must be specified "
        "using -input-image-list-file option.");
    parseInputImageList(inputImageListFile);
  }

  if (!expectedMatchingLabels.empty()) {
    // The number of category indices must match the number of files.
    if (expectedMatchingLabels.size() != inputImageFilenames.size()) {
      llvm::errs() << "Number of matching indices: "
                   << expectedMatchingLabels.size()
                   << " doesn't match the number of files: "
                   << inputImageFilenames.size() << "\n";
      return 1;
    }
  }

  // Stream input mode.
  const bool streamInputFilenamesMode =
      inputImageFilenames.size() == 1 && inputImageFilenames.front() == "-";

  GLOW_ASSERT(!(streamInputFilenamesMode && emittingBundle()) &&
              "Cannot emit a bundle and also stream inputs.");

  // Mini-batch mode.
  const bool miniBatchMode = miniBatch > 0;
  GLOW_ASSERT(((!miniBatchMode) || (!streamInputFilenamesMode)) &&
              "The minibatch option is not compatible with the stream input "
              "image mode.");
  GLOW_ASSERT(
      ((!miniBatchMode) || (inputImageFilenames.size() % miniBatch == 0)) &&
      "The number of input images must be a multiple of the mini-batch.");

  // Print out the inferred image classification.
  llvm::outs() << "Model: " << Loader::getModelOptPath() << "\n";
  std::mutex io_mu;
  int numErrors = 0;
  // Process a set of minibatches with indices [startIndex, endIndex).
  auto processImageRange = [&](size_t startIndex, size_t endIndex) {
    PlaceholderBindings bindings;
    Loader loader;
    // Used to make sure we only compile once, and run only once if not
    // streaming.
    bool isFirstRun = true;
    // These will be set during the first run.
    Placeholder *inputImagePH = nullptr;
    Tensor *SMT = nullptr;
    size_t miniBatchIndex = startIndex;
    Tensor inputImageData;
    std::vector<std::string> inputImageBatchFilenames;
    if ((!miniBatchMode) && (!streamInputFilenamesMode)) {
      inputImageBatchFilenames = inputImageFilenames;
    }

    while ((streamInputFilenamesMode &&
            getNextImageFilenames(&inputImageBatchFilenames)) ||
           (miniBatch &&
            getNextMiniBatch(inputImageBatchFilenames, inputImageFilenames,
                             miniBatchIndex, miniBatch, endIndex)) ||
           isFirstRun) {
      // Load and process the image data into the inputImageData Tensor.
      loadImagesAndPreprocess(inputImageBatchFilenames, &inputImageData,
                              imageNormMode, imageChannelOrder, imageLayout);

      // If this is the first run, then we need to build and compile the model.
      if (isFirstRun) {
        isFirstRun = false;

        // Build and compile the graph, and then get back the input Placeholder
        // and output Softmax Tensor.
        std::pair<Placeholder *, Tensor *> inputOutputPair =
            buildAndCompileAndGetInAndOutPair(loader, bindings,
                                              &inputImageData.getType());

        // If in bundle mode, the bundle has been saved by the above call, so we
        // can safely return.
        if (emittingBundle()) {
          return;
        }

        inputImagePH = inputOutputPair.first;
        SMT = inputOutputPair.second;
      }
      assert(inputImagePH && SMT && "Input and output must be valid.");
      GLOW_ASSERT(inputImagePH->dims() == inputImageData.dims() &&
                  "New input shape does not match the compiled function.");

      // Convert the raw input to fp16. This must be done every time we get new
      // image data.
      if (convertInAndOutToFp16) {
        inputImageData.convertToType(ElemKind::Float16Ty);
      }

      // About to run inference, so update the input image Placeholder's backing
      // Tensor with inputImageData.
      updateInputPlaceholders(bindings, {inputImagePH}, {&inputImageData});

      // Perform the inference execution, updating SMT.
      auto batchSize = inputImageData.dims()[0];
      loader.runInference(bindings, batchSize);

      // Print the top-k results from the output Softmax tensor.
      {
        ScopedLock lock(io_mu);
        numErrors += processAndPrintResults(SMT, inputImageBatchFilenames);
      }
    }

    // If profiling, generate and serialize the quantization infos now that we
    // have run inference one or more times to gather the profile.
    if (profilingGraph()) {
      loader.generateAndSerializeQuantizationInfos(bindings);
    }
  };

  // We will force single-threaded execution if:
  // - Minibatch mode is disabled;
  // - We are going to emit bundle and do not do inference;
  // - We are collecting inference profile.
  // Otherwise, there can be several minibatches of equal size.
  const bool multiThreadingAllowed =
      miniBatchMode && !emittingBundle() && !profilingGraph();
  const size_t numBatches =
      miniBatchMode ? inputImageFilenames.size() / miniBatch : 1u;
  const size_t numThreads = multiThreadingAllowed
                                ? std::min(size_t(miniBatchThreads), numBatches)
                                : 1u;
  if (miniBatchThreads > 1 && !multiThreadingAllowed) {
    llvm::outs() << "WARNING: multi-threaded execution is not possible. Make "
                    "sure that minibatch size is specified and you are not "
                    "trying to dump profile or emit bundle.\n";
  }

  llvm::outs() << "Running " << numThreads << " thread(s).\n";
  std::vector<std::thread> threads(numThreads);
  const size_t miniBatchesPerThread =
      (numBatches + numThreads - 1) / numThreads;
  for (size_t i = 0; i < numThreads; i++) {
    size_t startIndex, endIndex;
    if (numThreads > 1) {
      startIndex = i * miniBatchesPerThread * miniBatch;
      endIndex = std::min((i + 1) * miniBatchesPerThread * miniBatch,
                          inputImageFilenames.size());
    } else {
      startIndex = 0;
      endIndex = inputImageFilenames.size();
    }
    auto worker = [&processImageRange, startIndex, endIndex]() {
      processImageRange(startIndex, endIndex);
    };
    threads.push_back(std::thread(worker));
  }

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  return numErrors;
}
