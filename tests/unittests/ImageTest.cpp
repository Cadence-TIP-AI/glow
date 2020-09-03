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

#include "glow/Base/Image.h"

#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <utility>

using namespace glow;

TEST(Image, readNonSquarePngImage) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor vgaTensor;
  bool loadSuccess =
      !readPngImage(&vgaTensor, "tests/images/other/vga_image.png", range);
  ASSERT_TRUE(loadSuccess);

  auto &type = vgaTensor.getType();
  auto shape = vgaTensor.dims();

  // The loaded image is a 3D HWC tensor
  ASSERT_EQ(ElemKind::FloatTy, type.getElementType());
  ASSERT_EQ(3, shape.size());
  ASSERT_EQ(480, shape[0]);
  ASSERT_EQ(640, shape[1]);
  ASSERT_EQ(3, shape[2]);
}

TEST(Image, readBadImages) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor tensor;
  bool loadSuccess =
      !readPngImage(&tensor, "tests/images/other/dog_corrupt.png", range);
  ASSERT_FALSE(loadSuccess);

  loadSuccess =
      !readPngImage(&tensor, "tests/images/other/ghost_missing.png", range);
  ASSERT_FALSE(loadSuccess);
}

TEST(Image, readPngImageAndPreprocessWithAndWithoutInputTensor) {
  auto image1 = readPngImageAndPreprocess(
      "tests/images/imagenet/cat_285.png", ImageNormalizationMode::k0to1,
      ImageChannelOrder::RGB, ImageLayout::NHWC, imagenetNormMean,
      imagenetNormStd);
  Tensor image2;
  readPngImageAndPreprocess(image2, "tests/images/imagenet/cat_285.png",
                            ImageNormalizationMode::k0to1,
                            ImageChannelOrder::BGR, ImageLayout::NCHW,
                            imagenetNormMean, imagenetNormStd);

  // Test if the preprocess actually happened.
  dim_t imgHeight = image1.dims()[0];
  dim_t imgWidth = image1.dims()[1];
  dim_t numChannels = image1.dims()[2];

  Tensor transposed;
  image2.transpose(&transposed, {1u, 2u, 0u});
  image2 = std::move(transposed);

  Tensor swizzled(image1.getType());
  auto IH = image1.getHandle();
  auto SH = swizzled.getHandle();
  for (dim_t z = 0; z < numChannels; z++) {
    for (dim_t y = 0; y < imgHeight; y++) {
      for (dim_t x = 0; x < imgWidth; x++) {
        SH.at({x, y, numChannels - 1 - z}) = IH.at({x, y, z});
      }
    }
  }
  image1 = std::move(swizzled);
  EXPECT_TRUE(image1.isEqual(image2, 0.01));
}

TEST(Image, writePngImage) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor localCopy;
  bool loadSuccess =
      !readPngImage(&localCopy, "tests/images/imagenet/cat_285.png", range);
  ASSERT_TRUE(loadSuccess);

  llvm::SmallVector<char, 10> resultPath;
  llvm::sys::fs::createTemporaryFile("prefix", "suffix", resultPath);
  std::string outfilename(resultPath.begin(), resultPath.end());

  bool storeSuccess = !writePngImage(&localCopy, outfilename.c_str(), range);
  ASSERT_TRUE(storeSuccess);

  Tensor secondLocalCopy;
  loadSuccess = !readPngImage(&secondLocalCopy, outfilename.c_str(), range);
  ASSERT_TRUE(loadSuccess);
  EXPECT_TRUE(secondLocalCopy.isEqual(localCopy, 0.01));

  // Delete the temporary file.
  std::remove(outfilename.c_str());
}

TEST(Image, readMultipleInputsOpt) {
  imageLayoutOpt = {ImageLayout::NCHW, ImageLayout::NCHW};
  meanValuesOpt = {{127.5, 127.5, 127.5}, {0, 0, 0}};
  stddevValuesOpt = {{2, 2, 2}, {1, 1, 1}};
  imageChannelOrderOpt = {ImageChannelOrder::RGB, ImageChannelOrder::RGB};
  imageNormModeOpt = {ImageNormalizationMode::k0to255,
                      ImageNormalizationMode::k0to255};

  std::vector<std::vector<std::string>> filenamesList = {
      {"tests/images/imagenet/cat_285.png"},
      {"tests/images/imagenet/cat_285.png"}};
  Tensor image1;
  Tensor image2;
  loadImagesAndPreprocess(filenamesList, {&image1, &image2});

  auto H1 = image1.getHandle();
  auto H2 = image2.getHandle();
  EXPECT_EQ(H1.size(), H2.size());
  for (dim_t i = 0; i < H1.size(); i++) {
    EXPECT_FLOAT_EQ((H2.raw(i) - 127.5) / 2, H1.raw(i));
  }
}

TEST(Image, readMultipleInputsApi) {
  std::vector<ImageLayout> layout = {ImageLayout::NHWC, ImageLayout::NHWC};
  std::vector<std::vector<float>> mean = {{100, 100, 100}, {0, 0, 0}};
  std::vector<std::vector<float>> stddev = {{1.5, 1.5, 1.5}, {1, 1, 1}};
  std::vector<ImageChannelOrder> chOrder = {ImageChannelOrder::BGR,
                                            ImageChannelOrder::BGR};
  std::vector<ImageNormalizationMode> norm = {ImageNormalizationMode::k0to1,
                                              ImageNormalizationMode::k0to1};

  std::vector<std::vector<std::string>> filenamesList = {
      {"tests/images/imagenet/cat_285.png"},
      {"tests/images/imagenet/cat_285.png"}};
  Tensor image1;
  Tensor image2;
  loadImagesAndPreprocess(filenamesList, {&image1, &image2}, norm, chOrder,
                          layout, mean, stddev);

  auto H1 = image1.getHandle();
  auto H2 = image2.getHandle();
  EXPECT_EQ(H1.size(), H2.size());
  for (dim_t i = 0; i < H1.size(); i++) {
    EXPECT_NEAR((H2.raw(i) - (100 / 255.)) / 1.5, H1.raw(i), 0.0000001);
  }
}

/// Test writing a png image along with using the standard Imagenet
/// normalization when reading the image.
TEST(Image, writePngImageWithImagenetNormalization) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor localCopy;
  bool loadSuccess =
      !readPngImage(&localCopy, "tests/images/imagenet/cat_285.png", range,
                    imagenetNormMean, imagenetNormStd);
  ASSERT_TRUE(loadSuccess);

  llvm::SmallVector<char, 10> resultPath;
  llvm::sys::fs::createTemporaryFile("prefix", "suffix", resultPath);
  std::string outfilename(resultPath.begin(), resultPath.end());

  bool storeSuccess = !writePngImage(&localCopy, outfilename.c_str(), range,
                                     imagenetNormMean, imagenetNormStd);
  ASSERT_TRUE(storeSuccess);

  Tensor secondLocalCopy;
  loadSuccess = !readPngImage(&secondLocalCopy, outfilename.c_str(), range,
                              imagenetNormMean, imagenetNormStd);
  ASSERT_TRUE(loadSuccess);
  EXPECT_TRUE(secondLocalCopy.isEqual(localCopy, 0.02));

  // Delete the temporary file.
  std::remove(outfilename.c_str());
}
