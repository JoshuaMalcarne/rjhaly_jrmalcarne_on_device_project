## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

# Pyrealsense example code used to get started with sensor:  https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py
# This article helped with deployment  and postprocessing:  https://foundationsofdl.com/2021/07/20/depth-estimation-and-3d-mapping/

# First import the library
import pyrealsense2 as rs
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import os
import urllib
import tensorflow as tf
from PIL import Image


# Preprocesses the image to be input for tflite model
def preprocess_image(rgb_img, resize_shape):
    # Resize the image to match the model's input size
    resized_image = tf.image.resize(rgb_img, resize_shape)

    # Normalize the image
    normalized_image = resized_image / 255.0

    # Convert into tensor
    input_tensor = tf.expand_dims(normalized_image, axis=0)

    return input_tensor


# Postprocesses estimated depth image to original size
def postprocess_inference(img, module_output):
    # Obtain original image dimensions
    origin_height, origin_width, _ = img.shape
    im_pil = Image.fromarray(img)

    # Normalize the dephth image
    depth_min = module_output.min()
    depth_max = module_output.max()
    normalized_depth = (module_output - depth_min) / (depth_max - depth_min)

    # Uninverse the inverse depth map
    reversed_depth = 1.0 - normalized_depth

    # Apply colormap and resize image
    colored_depth = plt.cm.viridis(reversed_depth)
    colored_depth_squeezed = np.squeeze(colored_depth, axis=2)
    colored_depth_bgr = cv2.cvtColor((colored_depth_squeezed[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    colored_depth_resized = cv2.resize(colored_depth_bgr, (origin_width, origin_height), interpolation=cv2.INTER_CUBIC)

    # save estimated depth image
    cv2.imwrite("ece_579_images/estimated_depth.png", colored_depth_resized)

    # Overlap Estimated depth onto rgb image
    depth_rescaled = (255 * (module_output - depth_min) / (depth_max - depth_min)).astype("uint8")
    depth_rescaled_3chn = cv2.cvtColor(depth_rescaled,
                                       cv2.COLOR_GRAY2RGB)
    module_output_3chn = cv2.applyColorMap(depth_rescaled_3chn,
                                           cv2.COLORMAP_RAINBOW)
    module_output_3chn = cv2.resize(module_output_3chn,
                                    (origin_width, origin_height),
                                    interpolation=cv2.INTER_CUBIC)
    seg_pil = Image.fromarray(module_output_3chn.astype('uint8'), 'RGB')
    overlap = Image.blend(im_pil, seg_pil, alpha=0.6)

    return overlap


try:
    print("Loading model")
    # Download the tflite model if not currently downloaded.  Change this if running pruned model
    TFLITE_FILE_PATH = 'model/midas_v2_1_small.tflite'
    # TFLITE_FILE_PATH = 'model/midas_v2_1_smaller.tflite'

    if not os.path.isfile(TFLITE_FILE_PATH):
        tflite_model_url = "https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1?lite-format=tflite"
        urllib.request.urlretrieve(tflite_model_url, TFLITE_FILE_PATH)

    # Create image directory if one does not already exist
    image_dir = "ece_579_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Create interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
    interpreter.allocate_tensors()

    # 3: get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Configure realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Align depth and RGB images
    align = rs.align(rs.stream.color)

    num_frames = 3  # Number of frames for recording session.  Update based on required # of frames

    total_rmse = 0.0
    total_mae = 0.0
    total_inference_time = 0.0
    input_size = (256, 256)

    for i in range(num_frames):
        # Get frames from the pipeline
        frames = pipeline.wait_for_frames()

        # Align the color frame to the depth frame
        aligned_frames = align.process(frames)

        # Get individual aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Retrieve data from frames
        color_data = np.asanyarray(color_frame.get_data())
        depth_data = np.asanyarray(depth_frame.get_data())

        # normalize ground truth depth and save both images
        normalized_depth = (depth_data - np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))
        colored_depth = plt.cm.viridis(normalized_depth)
        colored_depth_bgr = cv2.cvtColor((colored_depth[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'ece_579_images/depth_image.png', colored_depth_bgr)
        cv2.imwrite('ece_579_images/color_image.png', color_data)

        # Resize ground truth for accuracy calculation
        resized_truth = cv2.resize(depth_data, input_size, interpolation=cv2.INTER_CUBIC)

        # Preprocess rgb image
        input_data = preprocess_image(color_data, input_size)

        # Start inference timer
        start_time = time.time()

        # Complete inference using preprocessed input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.squeeze(output_data, axis=0)

        # End inference timer
        end_time = time.time()

        # Postprocess the estimated depth and save overlap
        processed_data = postprocess_inference(color_data, output_data)
        processed_data.save('ece_579_images/overlap.png')

        # Reshape output data
        output_data = output_data[:, :, 0]

        # Scale the output data to match the units of measurement of L515: 1/4mm?
        output_data = output_data * 4

        # Create a mask for zero values in the ground truth image
        zero_mask = resized_truth == 0
        mask_uint8 = zero_mask.astype(np.uint8) * 255

        # Save mask as image
        cv2.imwrite('ece_579_images/mask_image.png', mask_uint8)

        # Calculate RMSE and MAE between ground truth and estimated depth. Ignore masked pixels
        rmse = np.sqrt(np.mean(((output_data - resized_truth) ** 2)[~zero_mask]))
        mae = np.mean(np.abs(output_data - resized_truth)[~zero_mask])
        total_rmse += rmse
        total_mae += mae

        # Calculate and print inference time
        inference_time = end_time - start_time
        total_inference_time += inference_time
        print(f"Frame {i + 1}: Inference Time - {inference_time:.4f} seconds, RMSE - {rmse:.4f}, MAE - {mae:.4f}")

    # Calculate and print average inference time, rmse, and mae
    average_inference_time = total_inference_time / num_frames
    average_rmse = total_rmse / num_frames
    average_mae = total_mae / num_frames

    print(f"Average Inference Time: {average_inference_time:.4f} seconds")
    print(f"Average RMSE: {average_rmse:.4f}")
    print(f"Average MAE: {average_mae:.4f}")

    # Stop pipeline
    pipeline.stop()
except rs.error as e:
    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
    #    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
    print("    %s\n", e.what())
    exit(1)
except Exception as e:
    print(e)
    pass