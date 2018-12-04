#include "caffe/layers/output_padding_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template <typename Dtype>
    void OutputPaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
        // Zero out all values in top layer
        caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());

        // Copy data from bottom to top
        for (int n = 0; n < NUM_; ++n) {
            for (int c = 0; c < CHANNELS_; ++c) {
                // Get GPU memory pointers to beginning of 2D tensor
                Dtype* dst = top[0]->mutable_gpu_data() + top[0]->offset(n, c);
                const Dtype* src = bottom[0]->gpu_data() + bottom[0]->offset(n, c);

                // Perform 2D CUDA memcopy
                CUDA_CHECK(cudaMemcpy2D(
                        dst,                          /* Destination start */
                        sizeof(Dtype) * WIDTH_OUT_,   /* Destination pitch (size of each row in bytes) */
                        src,                          /* Source start */
                        sizeof(Dtype) * WIDTH_IN_,    /* Source pitch (size of each row in bytes) */
                        sizeof(Dtype) * WIDTH_IN_,    /* Width to copy (as columns in bytes) */
                        HEIGHT_IN_,                   /* Height to copy (as number of rows) */
                        cudaMemcpyDeviceToDevice));
                }
        }
    }

    template <typename Dtype>
    void OutputPaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down,
                                                 const vector<Blob<Dtype>*>& bottom) {
        // Zero out bottom layer
        caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());

        // Copy diff from top to bottom
        for (int n = 0; n < NUM_; ++n) {
            for (int c = 0; c < CHANNELS_; ++c) {
                // Get GPU memory pointers to beginning of 2D tensor
                Dtype* dst = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n, c);
                const Dtype* src = top[0]->gpu_diff() + top[0]->offset(n, c);

                // Perform 2D CUDA memcopy
                CUDA_CHECK(cudaMemcpy2D(
                        dst,                          /* Destination start */
                        sizeof(Dtype) * WIDTH_IN_,    /* Destination pitch (size of each row in bytes) */
                        src,                          /* Source start */
                        sizeof(Dtype) * WIDTH_OUT_,   /* Source pitch (size of each row in bytes) */
                        sizeof(Dtype) * WIDTH_IN_,    /* Width to copy (as columns in bytes) */
                        HEIGHT_IN_,                   /* Height to copy (as number of rows) */
                        cudaMemcpyDeviceToDevice));
            }
        }
    }

    INSTANTIATE_LAYER_GPU_FUNCS(OutputPaddingLayer);
}
