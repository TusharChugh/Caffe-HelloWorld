#include "caffe/layers/channel_pad_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ChannelPadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  // Make sure that dimensions match
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  // Compute parameters and padding
  WIDTH_ = bottom[0]->width();
  HEIGHT_ = bottom[0]->height();
  NUM_ = bottom[0]->num();
  CHANNELS_IN_ = bottom[0]->channels();
  CHANNELS_OUT_ = bottom[1]->channels();
  PAD_ = CHANNELS_OUT_ - CHANNELS_IN_;

  // Reshape top layer to match shape of bottom 1
  top[0]->Reshape(NUM_, CHANNELS_OUT_, HEIGHT_, WIDTH_);
  CHECK_EQ(top[0]->count(), bottom[1]->count());
};

template <typename Dtype>
void ChannelPadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  // Zero all values in top layer
  caffe_set(top[0]->count(), Dtype(0), top_data);

  // Copy over relevant data
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNELS_IN_; ++c) {
        // Copy each channel to new padded position
        Dtype* destination = top_data + (n * CHANNELS_OUT_ + c + PAD_) * HEIGHT_ * WIDTH_;
        const Dtype* source = bottom_data + (n * CHANNELS_IN_ + c) * HEIGHT_ * WIDTH_;
        caffe_copy(HEIGHT_ * WIDTH_, source, destination);
    }
  }
}

template <typename Dtype>
void ChannelPadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  // Zero all values in bottom layer
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // Copy over relevant data
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNELS_IN_; ++c) {
        // Copy each channel from padded position in source to destination
        const Dtype* source = top_diff + (n * CHANNELS_OUT_ + c + PAD_) * HEIGHT_ * WIDTH_;
        Dtype* destination = bottom_diff + (n * CHANNELS_IN_ + c) * HEIGHT_ * WIDTH_;
        caffe_copy(HEIGHT_ * WIDTH_, source, destination);
    }
  }
}

INSTANTIATE_CLASS(ChannelPadLayer);
REGISTER_LAYER_CLASS(ChannelPad);

}  // namespace caffe
