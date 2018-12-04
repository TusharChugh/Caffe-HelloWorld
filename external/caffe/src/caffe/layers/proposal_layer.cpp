// ------------------------------------------------------------------
// Fast R-CNN
// Original work Copyright (c) 2015 Microsoft
// Modified work Copyright (c) 2017 AutoMap LLC / NextDroid LLC
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Mathias Bøgh Stokholm
// ------------------------------------------------------------------

#include "caffe/layers/proposal_layer.hpp"
#include "opencv2/opencv.hpp"

namespace caffe {
/**
 * @brief Simple struct used to simplify some of the proposal operations
 * @tparam Dtype: The type of storage to use for fields
 */
template <typename Dtype>
struct Anchor {
  Dtype x1;
  Dtype y1;
  Dtype x2;
  Dtype y2;
  Dtype score;

  // Comparison function to allow sorting Anchors by score field
  bool operator<(const Anchor<Dtype>& a) const {
    return score > a.score;
  }
};

/**
 * @brief Applies a set of predicted bounding box deltas to a set of root anchors in order to obtain a bounding box
 * @tparam Dtype: The type of storage to use for this operation
 * @param box [in]: The anchor that deltas should be applied to
 * @param delta [in]: The deltas to apply
 * @return The resulting bounding box
 */
template <typename Dtype>
cv::Vec<Dtype, 4> bboxTransformInv(const cv::Vec<Dtype, 4>& box, const cv::Vec<Dtype, 4>& delta) {
  Dtype src_w = box[2] - box[0] + 1;
  Dtype src_h = box[3] - box[1] + 1;
  Dtype src_ctr_x = (Dtype) (box[0] + 0.5 * src_w);
  Dtype src_ctr_y = (Dtype) (box[1] + 0.5 * src_h);
  Dtype pred_ctr_x = delta[0] * src_w + src_ctr_x;
  Dtype pred_ctr_y = delta[1] * src_h + src_ctr_y;
  Dtype pred_w = (Dtype) (exp(delta[2]) * src_w);
  Dtype pred_h = (Dtype) (exp(delta[3]) * src_h);
  return cv::Vec<Dtype, 4>(
      (pred_ctr_x - 0.5 * pred_w),
      (pred_ctr_y - 0.5 * pred_h),
      (pred_ctr_x + 0.5 * pred_w),
      (pred_ctr_y + 0.5 * pred_h)
  );
}

/**
 * @brief Clips a bounding box to fit inside a given rectangle
 * @tparam Dtype: The type of storage to use for this operation
 * @param box [in]: The bounding box that should be clipped
 * @param size [in]: The width and height of the rect to clip bounding box to
 * @return The clipped bounding box
 */
template <typename Dtype>
cv::Vec<Dtype, 4> clipBox(const cv::Vec<Dtype, 4>& box, const cv::Size_<Dtype>& size) {
  // Clamp box to image size
  return cv::Vec<Dtype, 4>(
      std::max(Dtype(0), box[0]),
      std::max(Dtype(0), box[1]),
      std::min(size.width - Dtype(1), box[2]),
      std::min(size.height - Dtype(1), box[3])
  );
}

/**
 * @brief Computes the Intersection-over-Union (IoU) of two Anchors/bounding boxes
 * @tparam Dtype: The type of storage to use for this operation
 * @param A [in]: The first Anchor/bounding box
 * @param B [in]: The second Anchor/bounding box
 * @return The computed overlap
 */
template <typename Dtype>
Dtype getIou(const Anchor<Dtype> &A, const Anchor<Dtype> &B) {
  const Dtype xx1 = std::max(A.x1, B.x1);
  const Dtype yy1 = std::max(A.y1, B.y1);
  const Dtype xx2 = std::min(A.x2, B.x2);
  const Dtype yy2 = std::min(A.y2, B.y2);
  Dtype inter = std::max(Dtype(0), xx2 - xx1 + 1) * std::max(Dtype(0), yy2 - yy1 + 1);
  Dtype areaA = (A.x2 - A.x1 + 1) * (A.y2 - A.y1 + 1);
  Dtype areaB = (B.x2 - B.x1 + 1) * (B.y2 - B.y1 + 1);
  return inter / (areaA + areaB - inter);
}

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  ProposalParam proposalParam = this->layer_param().proposal_param();

  // Extract base anchors to use
  std::copy(proposalParam.anchors().anchor().begin(), proposalParam.anchors().anchor().end(),
        std::back_inserter(_baseAnchors));

  // Get feature stride
  _feat_stride = proposalParam.feature_stride();

  // Decide whether to use training or testing values
  if (this->phase_ == TRAIN) {
    _rpn_pre_nms_top_n = proposalParam.rpn_pre_nms_top_n();
    _rpn_post_nms_top_n = proposalParam.rpn_post_nms_top_n();
    _rpn_nms_thresh = proposalParam.rpn_nms_thresh();
    _rpn_min_size = proposalParam.rpn_min_size();
  } else {
    _rpn_pre_nms_top_n = proposalParam.test_rpn_pre_nms_top_n();
    _rpn_post_nms_top_n = proposalParam.test_rpn_post_nms_top_n();
    _rpn_nms_thresh = proposalParam.test_rpn_nms_thresh();
    _rpn_min_size = proposalParam.test_rpn_min_size();
  }

  top[0]->Reshape(1, 5, 1, 1);
  if (top.size() > 1) {
    top[1]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  DLOG(ERROR) << "========== enter proposal layer";
  const Dtype *bottom_rpn_score = bottom[0]->cpu_data();  // rpn_cls_prob_reshape
  const Dtype *bottom_rpn_bbox = bottom[1]->cpu_data();   // rpn_bbox_pred
  const Dtype *bottom_im_info = bottom[2]->cpu_data();    // im_info

  const int num = bottom[1]->num();
  const int channes = bottom[1]->channels();
  const int height = bottom[1]->height();
  const int width = bottom[1]->width();
  CHECK(num == 1) << "only single item batches are supported";
  CHECK(channes % 4 == 0) << "rpn bbox pred channels should be divided by 4";

  const float im_height = bottom_im_info[0];
  const float im_width = bottom_im_info[1];

  const int config_n_anchors = _baseAnchors.size() / 4;
  LOG_IF(ERROR, _rpn_pre_nms_top_n <= 0 ) << "_rpn_pre_nms_top_n : " << _rpn_pre_nms_top_n;
  LOG_IF(ERROR, _rpn_post_nms_top_n <= 0 ) << "_rpn_post_nms_top_n : " << _rpn_post_nms_top_n;
  if (_rpn_pre_nms_top_n <= 0 || _rpn_post_nms_top_n <= 0 ) return;

  std::vector<Anchor<Dtype> > anchors;
  const Dtype min_size = bottom_im_info[2] * _rpn_min_size;

  DLOG(ERROR) << "========== generate anchors";

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      for (int k = 0; k < config_n_anchors; k++) {
        Dtype score = bottom_rpn_score[config_n_anchors * height * width +
                                       k * height * width + j * width + i];
        //const int index = i * height * config_n_anchors + j * config_n_anchors + k;

        cv::Vec<Dtype, 4> anchor(
            _baseAnchors[k * 4 + 0] + i * _feat_stride,  // shift_x[i][j];
            _baseAnchors[k * 4 + 1] + j * _feat_stride,  // shift_y[i][j];
            _baseAnchors[k * 4 + 2] + i * _feat_stride,  // shift_x[i][j];
            _baseAnchors[k * 4 + 3] + j * _feat_stride); // shift_y[i][j];

        cv::Vec<Dtype, 4> box_delta(
            bottom_rpn_bbox[(k * 4 + 0) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 4 + 1) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 4 + 2) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 4 + 3) * height * width + j * width + i]);

        cv::Vec<Dtype, 4> cbox = bboxTransformInv(anchor, box_delta);

        // 2. clip predicted boxes to image
        cbox = clipBox(cbox, cv::Size_<Dtype>(im_width, im_height));

        // 3. remove predicted boxes with either height or width < threshold
        if((cbox[2] - cbox[0] + 1) >= min_size && (cbox[3] - cbox[1] + 1) >= min_size) {
          Anchor<Dtype> anchor = {cbox[0], cbox[1], cbox[2], cbox[3], score};
          anchors.push_back(anchor);
        }
      }
    }
  }

  DLOG(ERROR) << "========== after clip and remove size < threshold box " << anchors.size();

  // Sort anchors according to score
  std::sort(anchors.begin(), anchors.end());

  // Limit number of anchors going into NMS
  int numAnchors = std::min((int) anchors.size(), _rpn_pre_nms_top_n);
  anchors.erase(anchors.begin() + numAnchors, anchors.end());

  // apply nms
  DLOG(ERROR) << "========== apply nms, pre nms number is : " << numAnchors;
  std::vector<bool> select(numAnchors, true);
  std::vector<cv::Vec<Dtype, 4> > box_final;
  std::vector<Dtype> scores_;

  // Perform non-maximum suppression
  for (unsigned int i = 0; i < numAnchors && box_final.size() < _rpn_post_nms_top_n; i++) {
    if (select[i]) {
      for (unsigned int j = i + 1; j < numAnchors; j++) {
        if (select[j]) {
          if (getIou(anchors[i], anchors[j]) >= _rpn_nms_thresh) {
            select[j] = false;
          }
        }
      }

      // Save anchor as accepted
      Anchor<Dtype> anchor = anchors[i];
      box_final.push_back(cv::Vec<Dtype, 4>(anchor.x1, anchor.y1, anchor.x2, anchor.y2));
      scores_.push_back(anchor.score);
    }
  }

  DLOG(ERROR) << "rpn number after nms: " <<  box_final.size();

  DLOG(ERROR) << "========== copy to top";
  top[0]->Reshape(box_final.size(), 5, 1, 1);
  Dtype *top_data = top[0]->mutable_cpu_data();
  CHECK_EQ(box_final.size(), scores_.size());
  for (size_t i = 0; i < box_final.size(); i++) {
    cv::Vec<Dtype, 4> &box = box_final[i];
    top_data[i * 5] = 0;
    for (int j = 1; j < 5; j++) {
      top_data[i * 5 + j] = box[j - 1];
    }
  }

  if (top.size() > 1) {
    top[1]->Reshape(box_final.size(), 1, 1, 1);
    for (size_t i = 0; i < box_final.size(); i++) {
      top[1]->mutable_cpu_data()[i] = scores_[i];
    }
  }

  DLOG(ERROR) << "========== exit proposal layer";
}

template <typename Dtype>
void ProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

} // namespace caffe
