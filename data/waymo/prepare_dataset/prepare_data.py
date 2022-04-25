# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import dataset_pb2
import matplotlib.pyplot as plt

from frame_utils import *
import numpy as np
import cv2
import os
import subprocess as sp

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import time

CAMERA_ID=0 # front camera

def rgba(r):
  """Generates a color based on range.

  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def plot_points_on_image(projected_points, camera_image, rgba_func,
                         point_size=5.0):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  plt.figure(figsize=(20, 12))
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.grid("off")

  xs = []
  ys = []
  colors = []

  for point in projected_points:
    xs.append(point[0])  # width, col
    ys.append(point[1])  # height, row
    colors.append(rgba_func(point[2]))

  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none") 
  plt.savefig('scattered.png')


def process_frame(frame, frame_path):
    '''
    Saves frames and the pointclouds of each frame. 
    We save the camera projected points with its depth values. 
    The projected points are save in a numpy binary float matrix.
    '''

    '''
    for i in range(len(frame.images)):
        print(i, dataset_pb2.CameraName.Name.Name(frame.images[i].name))
    '''


    img = tf.image.decode_jpeg(frame.images[CAMERA_ID].image)
    tf.io.write_file(frame_path + ".jpg", frame.images[CAMERA_ID].image)

    # read range image and camera projections
    (range_images, camera_projections, range_image_top_pose) = parse_range_image_and_camera_projection(frame)

    # convert range image to point cloud
    points, cp_points = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    # cp_pints[id][i] = {id?, w, h, 0, 0, 0}

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)

    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)

    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    # gathering the points in the coordinate sytem of the camera CAMERA_ID
    mask = tf.equal(cp_points_all_tensor[..., 0], frame.images[CAMERA_ID].name)

    cp_points_all_tensor = tf.cast(tf.gather_nd(
        cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    projected_points_all_from_raw_data = tf.concat(
        [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
        
    xy_depth = projected_points_all_from_raw_data[..., :3]
    np.save(frame_path + '_xy_depth.npy', xy_depth)

    xy_depth = np.load(frame_path + '_xy_depth.npy')
    h, w, c = img.shape
    gt = generate_depth_gt(xy_depth, h, w)
    debug_gt(gt, img.numpy(), frame_path + "_debug.jpg")

def debug_gt(depth, img, name):
    '''
    Args:
      depth: an array of shape [h, w]
      img: an array of shape [h, w, 3], intensities in range [0, 1]
    '''
    depth /= np.max(depth)
    depth = np.expand_dims(depth, 2)
    depth *= 3

    img = img.astype(np.float)/255
    img += depth
    img = np.clip(img, a_min=0.0, a_max=1.0)
    img *= 255
    img = img.astype(np.uint8)

    cv2.imwrite(name, img)

def generate_depth_gt(xy_depth, h, w):
    gt = np.zeros((h, w))
    xs = xy_depth[:, 1].astype(np.int)
    ys = xy_depth[:, 0].astype(np.int)
    gt[xs, ys] = xy_depth[:, 2]
    return gt

def process_sequence(filename, out_dir, num_frames=100):
  ''' Gets a frame sequence, intrinsic parameters, and point-cloud. 
  
  We use the FRONT camera (id = 1).
  Waymo has 31 tar files, each one has near to 25 sequence. We sample 100 frames by each sequence 
  by default. This are ~77K frames.

  Multiprocessing by image using TFRecordDataset map method

  It saves the sequences of frames, intrinsics, and points clouds.
  
  point_cloud on the world coordinate system: pc[i] = (x, y, X, Y, Z), (x, y) coordinates in the image plane, (X, Y, Z) coordinates in the world.
      we have to be able to read the point clouds in validation and test stages.
  '''

  # read tf record
  os.makedirs(out_dir, exist_ok=True)

  dataset = tf.data.TFRecordDataset(filename, compression_type='')

  intrinsics_set = set()
  for i, data in enumerate(dataset):
    if i >= num_frames:
        break

    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    intrinsics = [x for x in frame.context.camera_calibrations[0].intrinsic]
    intrinsics_set.add(tuple(intrinsics))

    process_frame(frame, os.path.join(out_dir, "{:06d}".format(i)))

    assert len(intrinsics_set) == 1
    np.savetxt(os.path.join(out_dir, 'intrinsics.txt'), intrinsics)

def process_dataset(dataset_dir, out_dir):
  # list tar files in training and validation folters
  tar_files = sp.run('ls {}/*/*.tar'.format(dataset_dir), shell=True, stdout=sp.PIPE)
  tar_files = tar_files.stdout.decode('utf-8').split('\n')

  # loop in tar file
  st = time.perf_counter()

  for tar_file in tar_files:
    if len(tar_file) == 0:
      continue

    tar_out_dir = os.path.join(out_dir, tar_file)
    if len(tar_file) > len(dataset_dir):
      if tar_file[:len(dataset_dir)] == dataset_dir:
        tar_out_dir = os.path.join(out_dir, tar_file[len(dataset_dir):].strip(os.sep))

    # untar file
    tar_out_dir = tar_out_dir[:-4]
    os.makedirs(tar_out_dir, exist_ok=True)
    _ = sp.run('tar -xf {} -C {}'.format(tar_file, tar_out_dir), shell=True, stdout=sp.PIPE)

    # list tfrecord files
    record_files = sp.run('ls {}/*.tfrecord'.format(tar_out_dir), shell=True, stdout=sp.PIPE)
    record_files = record_files.stdout.decode('utf-8').split('\n')

    # process sequences
    for record_file in record_files:
      if len(record_file) > 0:
        record_out_dir = record_file[:-9]
        os.makedirs(record_out_dir, exist_ok=True)
        
        process_sequence(record_file, record_out_dir)

    # delete all tf record file and tar file
    rm_cmd = 'ls {}/*.tfrecord | parallel \'rm {{}}\''.format(tar_out_dir)
    _ = sp.run(rm_cmd, shell=True, stdout=sp.PIPE)

    print("processed {} in {} mins".format(tar_file, (time.perf_counter() - st) // 60))
    st = time.perf_counter()


if __name__ == '__main__':
    dataset_dir = '/data/ra153646/datasets/waymo/waymo_test'
    out_dir = '/data/ra153646/datasets/waymo/waymo_test_processed'

    process_dataset(dataset_dir, out_dir)
    

