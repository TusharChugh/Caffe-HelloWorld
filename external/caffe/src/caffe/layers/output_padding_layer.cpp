#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/output_padding_layer.hpp"

namespace caffe {

    template <typename Dtype>
    void OutputPaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        PAD_H_ = this->layer_param().output_padding_param().pad_h();
        PAD_W_ = this->layer_param().output_padding_param().pad_w();

        NUM_ = bottom[0]->num();
        CHANNELS_ = bottom[0]->channels();
        WIDTH_IN_ = bottom[0]->width();
        HEIGHT_IN_ = bottom[0]->height();

        WIDTH_OUT_ = WIDTH_IN_ + PAD_W_;
        HEIGHT_OUT_ = HEIGHT_IN_ + PAD_H_;

        top[0]->Reshape(NUM_, CHANNELS_, HEIGHT_OUT_, WIDTH_OUT_);
        // TODO: Add checks?
    }

    template <typename Dtype>
    void OutputPaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                                const vector<Blob<Dtype> *> &top) {
        Dtype* top_data = top[0]->mutable_cpu_data();
        const Dtype* bottom_data = bottom[0]->cpu_data();

        // Zero out top
        caffe_set(top[0]->count(), Dtype(0), top_data);

        // Copy over data
        for (int n = 0; n < NUM_; ++n) {
            for (int c = 0; c < CHANNELS_; ++c) {
                for (int h = 0; h < HEIGHT_IN_; ++h) {
                    Dtype *destination = top_data + ((n * CHANNELS_ + c) * HEIGHT_OUT_ + h) * WIDTH_OUT_;
                    const Dtype *source = bottom_data + ((n * CHANNELS_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_;
                    caffe_copy(WIDTH_IN_, source, destination);
                }
            }
        }
    }

    template <typename Dtype>
    void OutputPaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down,
                                                 const vector<Blob<Dtype>*>& bottom) {

        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

        // Zero out bottom diff
        caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

        for (int n = 0; n < NUM_; ++n) {
            for (int c = 0; c < CHANNELS_; ++c) {
                for (int h = 0; h < HEIGHT_IN_; ++h) {
                    const Dtype* source = top_diff + ((n * CHANNELS_ + c) * HEIGHT_OUT_ + h) * WIDTH_OUT_;
                    Dtype *destination = bottom_diff + ((n * CHANNELS_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_;
                    caffe_copy(WIDTH_IN_, source, destination);
                }
            }
        }
    }

    INSTANTIATE_CLASS(OutputPaddingLayer);
    REGISTER_LAYER_CLASS(OutputPadding);
}