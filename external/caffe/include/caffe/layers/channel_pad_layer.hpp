#ifndef CAFFE_CHANNEL_PAD_LAYER_HPP_
#define CAFFE_CHANNEL_PAD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Layer for padding (with zeroes) the channels of an input layer to match
 *        those of another layer.
 *
 * Uses two bottom layers:
 *      0. The actual data layer that should be padded
 *      1. The layer whose size should be matched through padding of the channel dimension
 *
 * The top layer will have the same dimension as the bottom 1 layer, with the data
 * from the bottom 0 layer offset by the computed padding
 */
template <typename Dtype>
class ChannelPadLayer: public Layer<Dtype> {
 public:
  /**
   * @brief ChannelPadLayer constructor - doesn't use any layer parameters
   */
  explicit ChannelPadLayer(const LayerParameter& param): Layer<Dtype>(param) {}

  /**
   * @brief Computes the needed channel padding and reshapes the top layer to match
   *        the size of the second bottom (size reference layer)
   */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ChannelPad"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  private:
   int HEIGHT_;
   int WIDTH_;
   int NUM_;
   int CHANNELS_IN_;
   int CHANNELS_OUT_;
   int PAD_;
};

}  // namespace caffe

#endif  // CAFFE_CHANNEL_PAD_LAYER_HPP_
