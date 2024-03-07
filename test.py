import tensorflow as tf
import numpy as np


def assign_criterion_wh(gt_wh, anchors, anchor_threshold):
        # return: please note that the v5 default anchor_threshold is 4.0, related to the positive sample augment
        gt_wh = tf.expand_dims(gt_wh, 0)  # => 1 * n_gt * 2
        anchors = tf.expand_dims(anchors, 1)  # => n_anchor * 1 * 2
        ratio = gt_wh / anchors  # => n_anchor * n_gt * 2
        matched_matrix = tf.reduce_max(tf.math.maximum(ratio, 1 / ratio),
                                       axis=2) < anchor_threshold  # => n_anchor * n_gt
        return matched_matrix


def enrich_pos_by_position(assigned_label, assigned_anchor, gain, matched_matrix, rect_style='rect4'):
        # using offset to extend more postive result, if x
        assigned_xy = assigned_label[..., 0:2]  # n_matched * 2
        offset = tf.constant([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], tf.float32)
        grid_offset = tf.zeros_like(assigned_xy)

        if rect_style == 'rect2':
            g = 0.2  # offset
        elif rect_style == 'rect4':
            g = 0.5  # offset
            matched = (assigned_xy % 1. < g) & (assigned_xy > 1.)
            matched_left = matched[:, 0]
            matched_up = matched[:, 1]
            matched = (assigned_xy % 1. > (1 - g)) & (assigned_xy < tf.expand_dims(gain[0:2], 0) - 1.)
            matched_right = matched[:, 0]
            matched_down = matched[:, 1]

            assigned_anchor = tf.concat([assigned_anchor, assigned_anchor[matched_left], assigned_anchor[matched_up],
                                         assigned_anchor[matched_right], assigned_anchor[matched_down]], axis=0)
            assigned_label = tf.concat([assigned_label, assigned_label[matched_left], assigned_label[matched_up],
                                        assigned_label[matched_right], assigned_label[matched_down]], axis=0)

            grid_offset = g * tf.concat(
                [grid_offset, grid_offset[matched_left] + offset[1], grid_offset[matched_up] + offset[2],
                 grid_offset[matched_right] + offset[3], grid_offset[matched_down] + offset[4]], axis=0)

        return assigned_label, assigned_anchor, grid_offset

anchors = tf.constant([[[10,13], [16,30], [33,23]],
           [[30,61], [62,45], [59,119]],
           [[116,90], [156,198], [373,326]]], dtype=tf.float32)



label = [[0.22, 0.33, 0.24, 0.4, 0], 
         [0.24, 0.36, 0.74, 0.4, 1]]

img_size = 320
num_scales = 3
n_anchor_per_scale = 3
y_anchor_encode = []
gain = tf.ones(5, tf.float32)

grid_size = 80
y_true = tf.zeros([grid_size, grid_size, n_anchor_per_scale, 6], tf.float32)
gain = tf.tensor_scatter_nd_update(gain, [[0], [1], [2], [3]], [grid_size] * 4)

scaled_labels = label * gain
gt_wh = scaled_labels[..., 2:4]
matched_matrix = assign_criterion_wh(gt_wh, anchors[0], 4)

n_gt = tf.shape(gt_wh)[0]
a = tf.range(n_anchor_per_scale)
b = tf.reshape(a, (n_anchor_per_scale, 1))
c = tf.tile(b, (1, n_gt))

assigned_anchor = tf.expand_dims(c[matched_matrix], 1)
assigned_anchor = tf.cast(assigned_anchor, tf.int32)
# print(assigned_anchor)

assigned_label = tf.tile(tf.expand_dims(scaled_labels, 0), [n_anchor_per_scale, 1, 1])
assigned_label = assigned_label[matched_matrix]

# assigned_label, assigned_anchor, grid_offset = enrich_pos_by_position(
#                         assigned_label, assigned_anchor, gain, matched_matrix)

# print(assigned_label)
grid_offset = tf.zeros_like(assigned_label[:, 0:2])
assigned_grid = tf.cast(assigned_label[..., 0:2] - grid_offset, tf.int32)  # n_matched * 2
assigned_grid = tf.clip_by_value(assigned_grid, clip_value_min=0, clip_value_max=grid_size-1)

# tensor: grid * grid * 3 * 6, indices（sparse index）: ~n_gt * gr * gr * 3, updates: ~n_gt * 6
assigned_indices = tf.concat([assigned_grid[:, 1:2], assigned_grid[:, 0:1], assigned_anchor],
                                axis=1)

# print(assigned_indices)

xy, wh, clss = tf.split(assigned_label, (2, 2, 1), axis=-1)
xy = xy / gain[0] * img_size
wh = wh / gain[1] * img_size
obj = tf.ones_like(clss)
assigned_updates = tf.concat([xy, wh, obj, clss], axis=-1)

# print(assigned_updates)

y_true = tf.tensor_scatter_nd_update(y_true, assigned_indices, assigned_updates)
print(y_true.shape)
print(y_true[28][19])