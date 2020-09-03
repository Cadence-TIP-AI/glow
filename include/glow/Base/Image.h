/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
#ifndef GLOW_BASE_IMAGE_H
#define GLOW_BASE_IMAGE_H

#ifndef WITH_PNG
#error "Using Glow's PNG library requires installing libpng"
#endif

#include "glow/Base/Tensor.h"
#include "glow/Base/Type.h"
#include "glow/Support/Support.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <tuple>

namespace glow {

/// Pixel value ranges.
enum class ImageNormalizationMode {
  kneg1to1,     // Values are in the range: -1 and 1.
  k0to1,        // Values are in the range: 0 and 1.
  k0to255,      // Values are in the range: 0 and 255.
  kneg128to127, // Values are in the range: -128 .. 127
};

/// Layout of image dimensions (batch, channels, height, width).
enum class ImageLayout {
  NCHW,
  NHWC,
};

/// Order of color channels (red, green, blue).
enum class ImageChannelOrder {
  BGR,
  RGB,
};

/// All the image options are given as vectors, containing one element per model
/// input. An element at position i refers to input i, and input i refers to the
/// model input name given at the ith postion of the -model-input-name list.

/// NOTE: LLVM cmd parser made subclasses final in 3.7 yet the only cmd line
/// manual still refers to the old data and the change was not clear why it's
/// made. Assigning callbacks is not possible, and subclassing basic_parser is
/// open to future errors. Thus, relying in LLVM parser is minimized - we will
/// just obtain strings and process options. With the lack of Image class/struct
/// in Glow, we will have most of APIs to continue working with different APIs
/// directly affecting global cmd line arguments.

/// -image-mode flag.
extern std::vector<ImageNormalizationMode> imageNormModeOpt;

/// -image-channel-order flag.
extern std::vector<ImageChannelOrder> imageChannelOrderOpt;

/// -image-layout flag.
extern std::vector<ImageLayout> imageLayoutOpt;

/// -use-imagenet-normalization flag.
extern bool useImagenetNormalization;

/// -preprocessing parameters
extern VecVec<float> meanValuesOpt;
extern VecVec<float> stddevValuesOpt;

/// These are standard normalization factors for imagenet, adjusted for
/// normalizing values in the 0to255 range instead of 0to1, as seen at:
/// https://github.com/pytorch/examples/blob/master/imagenet/main.py
static const float imagenetNormMean[] = {0.485 * 255.0, 0.456 * 255.0,
                                         0.406 * 255.0};
static const float imagenetNormStd[] = {0.229, 0.224, 0.225};

/// Processes special command line args for Image module.
void processImageCmdArgVars(size_t numInputs);
/// Clear external storage for cmd args defined in Image.
void initImageCmdArgVars();

/// Default values for mean and stddev.
static const std::vector<float> zeroMean(max_tensor_dimensions, 0.f);
static const std::vector<float> oneStd(max_tensor_dimensions, 1.f);

/// \returns the floating-point range corresponding to enum value \p mode.
std::pair<float, float> normModeToRange(ImageNormalizationMode mode);

/// Reads a png image header from png file \p filename and \returns a tuple
/// containing height, width, and a bool if it is grayscale or not.
std::tuple<size_t, size_t, bool> getPngInfo(const char *filename);

/// Reads a png image. \returns True if an error occurred. The values of the
/// image are in the range \p range.
bool readPngImage(Tensor *T, const char *filename,
                  std::pair<float, float> range,
                  llvm::ArrayRef<float> mean = zeroMean,
                  llvm::ArrayRef<float> stddev = oneStd);

/// Writes a png image. \returns True if an error occurred. The values of the
/// image are in the range \p range.
bool writePngImage(Tensor *T, const char *filename,
                   std::pair<float, float> range,
                   llvm::ArrayRef<float> mean = zeroMean,
                   llvm::ArrayRef<float> stddev = oneStd);

/// Read a png image and preprocess it according to several parameters. Create a
/// tensor and store the preprocessed image data into this tensor.
/// \param filename the png file to read.
/// \param imageNormMode normalize values to this range.
/// \param imageChannelOrder the order of color channels.
/// \param imageLayout the order of dimensions (channel, height, and width).
/// \param mean use special mean to normalize.
/// \param stdev use special stddev to normalize.
Tensor readPngImageAndPreprocess(llvm::StringRef filename,
                                 ImageNormalizationMode imageNormMode,
                                 ImageChannelOrder imageChannelOrder,
                                 ImageLayout imageLayout,
                                 llvm::ArrayRef<float> mean = zeroMean,
                                 llvm::ArrayRef<float> stddev = oneStd);

/// Read a png image and preprocess it according to several parameters. Take a
/// tensor as a parameter and store the preprocessed image data into this
/// tensor.
/// \param imageData the tensor into which the preprocessed image data
///  will be stored.
/// \param filename the png file to read.
/// \param imageNormMode normalize values to this range.
/// \param imageChannelOrder the order of color channels.
/// \param imageLayout the order of dimensions (channel, height, and width).
/// \param mean use special mean to normalize.
/// \param stdev use special stddev to normalize.
void readPngImageAndPreprocess(Tensor &imageData, llvm::StringRef filename,
                               ImageNormalizationMode imageNormMode,
                               ImageChannelOrder imageChannelOrder,
                               ImageLayout imageLayout,
                               llvm::ArrayRef<float> mean = zeroMean,
                               llvm::ArrayRef<float> stddev = oneStd);

/// Loads either png or numpy images into the model input tensors.
/// \param filenamesList list of lists (for each input) of filenames to read.
/// \param inputImageDataList list of Tensors (for each input) that will
/// contain loaded and preprocessed images.
/// \param normMode normalize values to this range.
/// \param channelOrder the order of color channels.
/// \param imageLayout the order of dimensions (channel, height, and
/// width).
void loadImagesAndPreprocess(
    VecVecRef<std::string> filenamesList,
    llvm::ArrayRef<Tensor *> inputImageDataList,
    llvm::ArrayRef<ImageNormalizationMode> normMode = {},
    llvm::ArrayRef<ImageChannelOrder> channelOrder = {},
    llvm::ArrayRef<ImageLayout> imageLayout = {}, VecVecRef<float> mean = {},
    VecVecRef<float> stddev = {});

} // namespace glow

#endif // GLOW_BASE_IMAGE_H
