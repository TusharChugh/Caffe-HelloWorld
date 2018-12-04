#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cfloat>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/argmax_layer.hpp"

namespace caffe {

/**
 * @brief Finds and stores the indexes of the highest values along the first
 *        axis (channels).
 * @param n [in]: Number of threads to spawn
 * @param num [in]: The index of image to process from batch
 * @param channels [in]: The number of channels
 * @param width [in]: The width of each image
 * @param height [in]: The height of each image
 * @param src [in]: A pointer to the GPU source (input) data
 * @param dst [out]: A pointer to the GPU destination (output) data
 */
template <typename Dtype>
__global__ void kernel_channel_max_element(const int n, const int num, const int channels,
                                           const int width, const int height,
                                           const Dtype* src, Dtype* dst) {
  // Calculate the size of each channel
  const int channelSize = height * width;

  // Create a pointer to the first applicable element in the source data
  const Dtype* p = src + (num * channels) * channelSize;

  // Loop over each pixel and find maximum value along channel axis
  CUDA_KERNEL_LOOP(index, n) {
    // Determine index of maximum channel value
    int maxIndex = 0;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
      Dtype val = *(p + index + c * channelSize);
      if (val > maxval) {
        maxIndex = c;
        maxval = val;
      }
    }

    // Store resulting index
    dst[num * channels + index] = maxIndex;
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {

  // GPU implementation only supports ArgMax along axis 1 (channels). It doesn't
  // support more than one output value, or returning the highest element (instead of index)
  if (!has_axis_ || top_k_ != 1 || out_max_val_ || axis_ != 1) {
    // Fall back to CPU implementation
    return Forward_cpu(bottom, top);
  }

  // GPU implementation is supported
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = bottom[0]->count();
  int channels = bottom[0]->channels();
  int NUM_ = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  // Launch a set of kernels for each feature in batch
  for (int num = 0; num < NUM_; num++) {
    kernel_channel_max_element<<<CAFFE_GET_BLOCKS(width * height),
      CAFFE_CUDA_NUM_THREADS>>>(width * height, num, channels, width, height,
      bottom_data, top_data);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ArgMaxLayer);

}  // namespace caffe
