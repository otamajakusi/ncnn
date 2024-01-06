// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <map>

#include "mahjong.h"
#include "score.h"
#include "util.h"

ncnn::Net yolov5;

const int target_size = 640;
const float prob_threshold = 0.50f;
const float nms_threshold = 0.45f;
const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};


//#define YOLOV5_V60 1 //YOLOv5 v6.0
#define YOLOV5_V62 1 //YOLOv5 v6.2 export  onnx model method https://github.com/shaoshengsong/yolov5_62_export_ncnn

#if YOLOV5_V60 || YOLOV5_V62
#define MAX_STRIDE 64
#else
#define MAX_STRIDE 32
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)
#endif //YOLOV5_V60    YOLOV5_V62

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_confidence = sigmoid(featptr[4]);
                if (box_confidence >= prob_threshold)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);
                    if (confidence >= prob_threshold)
                    {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
{
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;

#if YOLOV5_V62
        ex.extract("353", out);
#elif YOLOV5_V60
        ex.extract("376", out);
#else
        ex.extract("781", out);
#endif

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
#if YOLOV5_V62
        ex.extract("367", out);
#elif YOLOV5_V60
        ex.extract("401", out);
#else
        ex.extract("801", out);
#endif
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    // https://github.com/otamajakusi/yolov5/blob/master/tiles.yaml
    static const char* class_names[] = {
        "bk",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
        "m8",
        "m9",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "sc",
        "sh",
        "sw",
        "wn",
        "wp",
        "ws",
        "wt"};

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //        obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(bgr, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        //sprintf(text, "%s %.1f", class_names[obj.label], obj.prob * 100);
        sprintf(text, "%s", class_names[obj.label]);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

const MJTileId label2TileId[] = { // yolov5Id -> mahjong-c Id
      // mahjong-c : yolov5
  MJTileId(-1), // "bk", -1, : //"bk", // 0
  MJ_M1, // "m1",  0, : //"m1", // 1
  MJ_M2, // "m2",  1  : //"m2", // 2
  MJ_M3, // "m3",  2  : //"m3", // 3
  MJ_M4, // "m4",  3  : //"m4", // 4
  MJ_M5, // "m5",  4  : //"m5", // 5
  MJ_M6, // "m6",  5  : //"m6", // 6
  MJ_M7, // "m7",  6  : //"m7", // 7
  MJ_M8, // "m8",  7  : //"m8", // 8
  MJ_M9, // "m9",  8  : //"m9", // 9
  MJ_P1, // "p1",  9  : //"p1", // 10
  MJ_P2, // "p2", 10  : //"p2", // 11
  MJ_P3, // "p3", 11  : //"p3", // 12
  MJ_P4, // "p4", 12  : //"p4", // 13
  MJ_P5, // "p5", 13  : //"p5", // 14
  MJ_P6, // "p6", 14  : //"p6", // 15
  MJ_P7, // "p7", 15  : //"p7", // 16
  MJ_P8, // "p8", 16  : //"p8", // 17
  MJ_P9, // "p9", 17  : //"p9", // 18
  MJ_S1, // "s1", 18  : //"s1", // 19
  MJ_S2, // "s2", 19  : //"s2", // 20
  MJ_S3, // "s3", 20  : //"s3", // 21
  MJ_S4, // "s4", 21  : //"s4", // 22
  MJ_S5, // "s5", 22  : //"s5", // 23
  MJ_S6, // "s6", 23  : //"s6", // 24
  MJ_S7, // "s7", 24  : //"s7", // 25
  MJ_S8, // "s8", 25  : //"s8", // 26
  MJ_S9, // "s9", 26  : //"s9", // 27
  MJ_DR, // "wt", 27  : //"dr", // 28
  MJ_DG, // "wn", 28  : //"dg", // 29
  MJ_DW, // "ws", 29  : //"dw", // 30
  MJ_WN, // "wp", 30  : //"wn", // 31
  MJ_WP, // "dw", 31  : //"wp", // 32
  MJ_WS, // "dg", 32  : //"ws", // 33
  MJ_WT, // "dr", 33  : //"wt", // 34
};

bool compObjectX(Object &o1, Object &o2) {
    return o1.rect.x < o2.rect.x;
}

bool getSortedAreaObjects(
    const std::vector<Object> &objects,
    const cv::Mat &m,
    std::vector<Object> &upper,
    std::vector<Object> &lower)
{
    int width = m.cols;
    int height = m.rows;
    for (auto obj : objects) {
        if ((obj.rect.y + obj.rect.height / 2) > height / 2) {
            lower.push_back(obj);
        } else {
            upper.push_back(obj);
        }
    }
    if (lower.empty()) {
        std::cout << "手の内に牌がありません" << std::endl;
        return false;
    }
    if (upper.empty()) {
        std::cout << "アガリ牌がありません" << std::endl;
        return false;
    }
    sort(lower.begin(), lower.end(), compObjectX);
    sort(upper.begin(), upper.end(), compObjectX);
    return true;
}

bool handleLower(
    const std::vector<Object> &lower,
    MJMelds &melds,
    MJHands &hands)
{
    for (int i = 0; i < lower.size(); i ++) {
        if (lower[i].label == 0/*bk*/) { // start concealed four check
            if (i + 4 > lower.size() ||
              (lower[i + 1].label != lower[i + 2].label) ||
              (lower[i + 3].label != 0/*bk*/)) {
                std::cout << "暗槓の指定が不正です" << std::endl;
                return false;
            }
	    if (melds.len >= 4) {
                std::cout << "手の内の数が不正です" << std::endl;
                return false;
	    }
            const MJTileId tileId = label2TileId[lower[i + 1].label];
	    MJMeld *meld = &melds.meld[melds.len];
	    meld->tile_id[0] = tileId;
	    meld->tile_id[1] = tileId;
	    meld->tile_id[2] = tileId;
	    meld->tile_id[3] = tileId;
	    meld->len = 4;
	    meld->concealed = true;
	    melds.len ++;
            hands.tile_id[hands.len ++] = tileId;
            hands.tile_id[hands.len ++] = tileId;
            hands.tile_id[hands.len ++] = tileId;
            hands.tile_id[hands.len ++] = tileId;
        } else {
            const MJTileId tileId = label2TileId[lower[i].label];
            hands.tile_id[hands.len ++] = tileId;
	}
    }
    if ((lower.size() - melds.len * 4 - 1) % 3 != 0) {
        std::cout << "手の内の数が不正です" << std::endl;
        return false;
    }
    return true;
}

// pickup num of tiles from upper from index
void pickup(
    const std::vector<Object> &upper,
    int num,
    MJMelds &melds,
    MJHands &hands,
    int &index,
    int &rest) {
    MJMeld *meld = &melds.meld[melds.len];
    int meld_index = 0;
    for (int i = 0; i < num; i ++) {
        MJTileId tileId = label2TileId[upper[index + i].label];
        hands.tile_id[hands.len ++] = tileId;
	meld->tile_id[meld_index ++] = tileId;
    }
    if (num > 1) {
      meld->len = num; // FIXME: 3 or 4 should be checked
      meld->concealed = false;
      melds.len ++;
    }
    index += num;
    rest -= num;
}

bool isSame(
    const std::vector<Object> &upper,
    int num, int index) {
    MJTileId tileId = label2TileId[upper[index].label];
    for (int i = 1; i < num; i ++) {
        if (tileId != label2TileId[upper[index + i].label]) {
            return false;
        }
    }
    return true;
}


bool handleUpper(
    const std::vector<Object> &upper,
    MJMelds &melds,
    MJHands &hands,
    int elemNum)
{
#define PICKUP(num)     pickup(upper, num, melds, hands, index, rest)
#define IS_SAME(num)    isSame(upper, num, index)
    // check if upper has 'bk'
    for (auto e : upper) {
        if (e.label == 0/*bk*/) {
            std::cout << "副露牌が不正です" << std::endl;
            return false;
        }
    }
    int index = 0;
    int rest = upper.size();
    int elemRest = elemNum;
    while (1) {
        if (elemRest == 0) {
            if (rest == 1) {
                PICKUP(1);
                return true;
            } else {
                std::cout << "副露牌が不正です" << std::endl;
                return false;
            }
        } else if (elemRest == 1) {
            if (rest == 3 + 1) {
                PICKUP(3);
            } else if (rest == 4 + 1) {
                PICKUP(4);
            } else {
                std::cout << "副露牌が不正です" << std::endl;
                return false;
            }
        } else if (elemRest == 2) {
            if (rest == 6 + 1) {
                PICKUP(3);
            } else if (rest == 7 + 1) {
                if (IS_SAME(4)) {
                    PICKUP(4);
                } else {
                    PICKUP(3);
                }
            } else if (rest == 8 + 1) {
                PICKUP(4);
            } else {
                std::cout << "副露牌が不正です" << std::endl;
                return false;
            }
        } else if (elemRest == 3) {
            if (rest == 9 + 1) {
                PICKUP(3);
            } else if (rest == 10 + 1 || rest == 11 + 1) {
                if (IS_SAME(4)) {
                    PICKUP(4);
                } else {
                    PICKUP(3);
                }
            } else if (rest == 12 + 1) {
                PICKUP(4);
            } else {
                std::cout << "副露牌が不正です" << std::endl;
                return false;
            }
        } else if (elemRest == 4) {
            if (rest == 12 + 1) {
                PICKUP(3);
            } else if (rest == 13 + 1 || rest == 14 + 1 || rest == 15 + 1) {
                if (IS_SAME(4)) {
                    PICKUP(4);
                } else {
                    PICKUP(3);
                }
            } else if (rest == 16 + 1) {
                PICKUP(4);
            } else {
                std::cout << "副露牌が不正です" << std::endl;
                return false;
            }
        } else {
            std::cout << "副露牌が不正です" << std::endl;
            return false;
        }
        elemRest --;
    }
}

std::map<std::string, cv::Mat> yakuImages = {
    {"pinfu", cv::imread("pinfu.png")},			// 平和
    {"tanyao", cv::imread("tanyao.png")},			// 断么九
    {"iipeiko", cv::imread("iipeiko.png")},		// 一盃口
    {"haku", cv::imread("haku.png")},			// 白
    {"hatsu", cv::imread("hatsu.png")},			// 發
    {"chun", cv::imread("chun.png")},			// 中
    {"ton", cv::imread("ton.png")},			// 東
    {"nan", cv::imread("nan.png")},			// 南
    {"sha", cv::imread("sha.png")},			// 西
    {"pei", cv::imread("pei.png")},			// 北
    {"tsumo", cv::imread("tsumo.png")},			// 門前清自摸和
    {"toitoi", cv::imread("toitoi.png")},			// 対々和
    {"sanankou", cv::imread("sanankou.png")},		// 三暗刻
    {"sanshoku_douko", cv::imread("sanshoku_douko.png")},	// 三色同刻 
    {"sankantsu", cv::imread("sankantsu.png")},		// 三槓子
    {"shosangen", cv::imread("shosangen.png")},		// 小三元
    {"honroto", cv::imread("honroto.png")},		// 混老頭
    {"double_ton", cv::imread("double_ton.png")},		// ダブ東
    {"double_nan", cv::imread("double_nan.png")},		// ダブ南
    {"double_sha", cv::imread("double_sha.png")},		// ダブ西
    {"double_pei", cv::imread("double_pei.png")},		// ダブ北
    {"chiitoitsu", cv::imread("chiitoitsu.png")},		// 七対子
    {"sanshoku", cv::imread("sanshoku.png")},		// 三色同順
    {"ittsu", cv::imread("ittsu.png")},			// 一気通貫
    {"chanta", cv::imread("chanta.png")},			// 混全帯么九
    {"ryanpeiko", cv::imread("ryanpeiko.png")},		// 二盃口
    {"honitsu", cv::imread("honitsu.png")},		// 混一色
    {"junchan", cv::imread("junchan.png")},		// 純全帯么九
    {"chinitsu", cv::imread("chinitsu.png")},		// 清一色
    {"kokushi", cv::imread("kokushi.png")},		// 国士無双
    {"suuankou", cv::imread("suuankou.png")},		// 四暗刻
    {"daisangen", cv::imread("daisangen.png")},		// 大三元
    {"ryuisou", cv::imread("ryuisou.png")},		// 緑一色
    {"tsuisou", cv::imread("tsuisou.png")},		// 字一色
    {"shosuushi", cv::imread("shosuushi.png")},		// 小四喜
    {"daisuushi", cv::imread("daisuushi.png")},		// 大四喜
    {"chinroto", cv::imread("chinroto.png")},		// 清老頭
    {"suukantsu", cv::imread("suukantsu.png")},		// 四槓子
    {"chuuren_poutou", cv::imread("chuuren_poutou.png")},	// 九蓮宝燈
};

bool calcMahjongScore(
    const std::vector<Object> &objects,
    const cv::Mat &m,
    MJTileId playerWind,
    MJTileId roundWind,
    bool tsumo)
{
    // 1. split objects into upper_objects and lower_objects
    // 2. if lower_objects is empty, illegal. at least hand tile is needed.
    // 3. if upper_objects is empty, illegal. at least win tile is needed.
    // 4. sort both lower and upper objects by x
    // 5. handle lower_objects
    // 5.1. pick up concealed four if `bk` is found
    // 6. handle upper_objects
    // 6.1. pick up win tile if `wn` is found
    bool ret;
    std::vector<Object> upper;
    std::vector<Object> lower;
    ret = getSortedAreaObjects(objects, m, upper, lower);
    if (!ret) {
        return ret;
    }
    MJMelds melds;
    MJHands hands;
    memset(&melds, 0, sizeof(melds));
    memset(&hands, 0, sizeof(hands));
    ret = handleLower(lower, melds, hands);
    if (!ret) {
	return ret;
    }
    const int lowerElemNum = (lower.size() - melds.len * 4 - 1) / 3 + melds.len;
    ret = handleUpper(upper, melds, hands, MJ_ELEMENTS_LEN - lowerElemNum);
    if (!ret) {
	return ret;
    }
    const MJTileId winTile = label2TileId[upper[upper.size()-1].label];
    MJBaseScore score;
    memset(&score, 0, sizeof(score));
    int mj_ret = mj_get_score(&score, &hands, &melds, winTile, !tsumo, playerWind, roundWind);
    if (mj_ret != MJ_OK) {
	std::cout << "mj_get_score " << mj_ret << std::endl;
	return false;
    }
    std::cout << "yaku: " << score.yaku_name << ", han:" << score.han << std::endl;

    uint32_t point;
    uint32_t pointDealer;
    bool dealer = playerWind == MJ_WT;
    get_score(score.fu, score.han, tsumo, dealer, &point, &pointDealer);
    std::cout << "point: " << point << ", pointDealer: " << pointDealer << std::endl;

    std::string pointStr;
    if (dealer) {
	if (tsumo) {
	    pointStr = std::to_string(point) + " ALL";
	} else {
	    pointStr = std::to_string(point);
	}
    } else {
	if (tsumo) {
	    pointStr = std::to_string(point) + "/" + std::to_string(pointDealer);
	} else {
	    pointStr = std::to_string(point);
	}
    }
    cv::putText(m, pointStr, cv::Point(64 * 3, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(10, 10, 10), 2);
    cv::putText(m, pointStr, cv::Point(64 * 3+1, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(250,250,250), 2);


    //cv::Mat &_yakuImg = yakuImages["pinfu"];
    //cv::Mat _yakuRoi = m(cv::Rect(0, 128, _yakuImg.cols, _yakuImg.rows));
    //_yakuImg.copyTo(yakuRoi);
    // yaku
    std::istringstream iss(score.yaku_name);
    std::vector<std::string> yaku;
    std::string sub;
    while (iss >> sub) {
	yaku.push_back(sub);
    }
    int yaku_x = 0;
    for (auto y : yaku) {
	std::cout << "yaku: " << y << std::endl;
	auto it = yakuImages.find(y);
	if (it == yakuImages.end()) {
	    std::cout << "yaku image not found: " << y << std::endl;
	    continue;
	}
	cv::Mat &yakuImg = it->second;
	cv::Mat yakuRoi = m(cv::Rect(yaku_x, m.size().height - yakuImg.rows, yakuImg.cols, yakuImg.rows));
	yakuImg.copyTo(yakuRoi);
	yaku_x += yakuImg.cols;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [video device number]\n", argv[0]);
        return -1;
    }

    int video_device = atoi(argv[1]);

    cv::Mat m;
    cv::VideoCapture cap(video_device, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        fprintf(stderr, "cap.isOpened %d failed\n", video_device);
        return -1;
    }

    // yolov5.opt.use_vulkan_compute = true;
    // yolov5.opt.use_bf16_storage = true;

    // original pretrained model from https://github.com/ultralytics/yolov5
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
#if YOLOV5_V62
    if (yolov5.load_param("yolov5s_6.2.mj.param"))
        exit(-1);
    if (yolov5.load_model("yolov5s_6.2.mj.bin"))
        exit(-1);
#elif YOLOV5_V60
    if (yolov5.load_param("yolov5s_6.0.param"))
        exit(-1);
    if (yolov5.load_model("yolov5s_6.0.bin"))
        exit(-1);
#else
    yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    if (yolov5.load_param("yolov5s.param"))
        exit(-1);
    if (yolov5.load_model("yolov5s.bin"))
        exit(-1);
#endif

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    int playerWindIndex = 0;
    int roundWindIndex = 0;
    int agariIndex = 0;

    cv::Mat playerWindImgs[] = {
	    cv::imread("ton-ke.png"),
	    cv::imread("nan-ke.png"),
	    cv::imread("sha-ke.png"),
	    cv::imread("pei-ke.png")};

    cv::Mat roundWindImgs[] = {
	    cv::imread("ton-ba.png"),
	    cv::imread("nan-ba.png"),
	    cv::imread("sha-ba.png"),
	    cv::imread("pei-ba.png")};

    cv::Mat agariImgs[] = {
	    cv::imread("tsumo-agari.png"),
	    cv::imread("ron-agari.png")};

    int loop = true;
    while (loop)
    {
        cap.read(m);
        if (m.empty())
        {
            fprintf(stderr, "End of stream\n");
            break;
        }

        std::vector<Object> objects;
        detect_yolov5(m, objects);
        draw_objects(m, objects);

	cv::Mat &playerWindImg = playerWindImgs[playerWindIndex];
	cv::Mat &roundWindImg = roundWindImgs[roundWindIndex];
	cv::Mat &agariImg = agariImgs[agariIndex];

	cv::Mat playerWindRoi = m(cv::Rect(0, 0, playerWindImg.cols, playerWindImg.rows));
	playerWindImg.copyTo(playerWindRoi);

	cv::Mat roundWindRoi = m(cv::Rect(playerWindImg.cols, 0, roundWindImg.cols, roundWindImg.rows));
	roundWindImg.copyTo(roundWindRoi);

	cv::Mat agariRoi = m(cv::Rect(playerWindImg.cols + roundWindImg.cols, 0, agariImg.cols, agariImg.rows));
	agariImg.copyTo(agariRoi);

        calcMahjongScore(objects, m, (MJTileId)(playerWindIndex + MJ_WT), (MJTileId)(roundWindIndex + MJ_WT), agariIndex == 0);

        frame_count++;
        total_frames++;

        if (frame_count >= 30)
        {

            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

#if 0
        if (fps > 0)
        {

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(m, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
#endif

        cv::imshow("yolov5_ncnn",m);
	int key = cv::waitKey(1);

	switch (key) {
	case 'q':
	case 27:
	    loop = false;
	    break;
	case '1':
	    playerWindIndex = (playerWindIndex + 1) % 4;
	    break;
	case '2':
	    roundWindIndex = (roundWindIndex + 1) % 4;
	    break;
	case '3':
	    agariIndex = (agariIndex + 1) % 2;
	    break;
	case '4':
	    break;
	default:
	    break;
        }
    }
    cap.release();
    fprintf(stderr, "finished by user\n");
    return 0;
}
