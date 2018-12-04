#include "caffe/layers/channel_pad_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Copies over each channel using a separate GPU thread
template <typename Dtype>
__global__ void copy_kernel(const int channels_in, const int channels_out,
    const int width, const int height, const int pad, const int num,
    const Dtype* src, Dtype* dst, bool forward) {

  // Loop over channels
  CUDA_KERNEL_LOOP(c, channels_in) {
    int dst_start;
    int src_start;

    // Check if doing a forward or backward pass
    if (forward) {
        dst_start = (num * channels_out + c + pad) * height * width;
        src_start = (num * channels_in + c) * height * width;
    } else {
        // Backward
        src_start = (num * channels_out + c + pad) * height * width;
        dst_start = (num * channels_in + c) * height * width;
    }

    // Do actual copy of feature
    for (int i = 0; i < width * height; ++i) {
      dst[dst_start + i] = src[src_start + i];
    }
  }
}

template <typename Dtype>
void ChannelPadLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  // Zero all values in top layer
  CUDA_CHECK(cudaMemset(top_data, 0, sizeof(Dtype) * top[0]->count()));

  // Launch a set of kernels for each feature in batch
  for (int num = 0; num < NUM_; num++) {
      copy_kernel<<<CAFFE_GET_BLOCKS(CHANNELS_IN_), CAFFE_CUDA_NUM_THREADS>>>(
          CHANNELS_IN_, CHANNELS_OUT_, WIDTH_, HEIGHT_, PAD_, num, bottom_data, top_data, true);
      CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void ChannelPadLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();

  // Zero all values in bottom layer
  CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count()));

  // Launch a set of kernels for each feature in batch
  for (int num = 0; num < NUM_; num++) {
      copy_kernel<<<CAFFE_GET_BLOCKS(CHANNELS_IN_), CAFFE_CUDA_NUM_THREADS>>>(
          CHANNELS_IN_, CHANNELS_OUT_, WIDTH_, HEIGHT_, PAD_, num, top_diff, bottom_diff, false);
      CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelPadLayer);

} // namespace caffe
