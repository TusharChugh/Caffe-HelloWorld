#ifndef CAFFE_OUTPUT_PADDING_LAYER_
#define CAFFE_OUTPUT_PADDING_LAYER_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

    template <typename Dtype>
    class OutputPaddingLayer : public Layer<Dtype> {
    public:
        /**
         * @brief Our Output Padding Layer that given a pad_h, pad_w we apply that to the output
         * @param param [in]: The layer parameters (pad_h, pad_w)
         */
        explicit OutputPaddingLayer(const LayerParameter& param):
                Layer<Dtype>(param) {}

        /**
         * @brief Layer setup is going to declare padding
         * @param bottom
         * @param top
         */
//        virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
//                                const vector<Blob<Dtype>*> &top);

        /**
         * @brief Reshapes the top layer to have padding
         * @param bottom [in]: The input data
         * @param top [in]: The output data
         */
        virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                             const vector<Blob<Dtype>*> &top);

        virtual inline const char* type() const { return "OutputPadding"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
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
        int PAD_H_;
        int PAD_W_;

        int NUM_;
        int CHANNELS_;
        int WIDTH_IN_;
        int HEIGHT_IN_;

        int WIDTH_OUT_;
        int HEIGHT_OUT_;
    };

}

#endif //CAFFE_OUTPUT_PADDING_LAYER_
