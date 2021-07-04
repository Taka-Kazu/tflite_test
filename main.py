import numpy as np
import tensorflow as tf
import cv2
import time
import argparse

def load_labels(path):
  labels = {}
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for _, num_name in enumerate(lines):
      num_name = num_name.split()
      num = num_name[0]
      name = num_name[1]
      labels[int(num)] = name
  return labels


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run tflite object detector')
  parser.add_argument('--video', type=str, default='video.mp4', help='video file or device (default: video.mp4)')
  parser.add_argument('--model', type=str, default='ssdlite_mobiledet_coco_qat_postprocess.tflite', help='tflite model')
  parser.add_argument('--label_file', type=str, default='coco_labels.txt', help='object detection label file (default: coco_labels.txt)')
  parser.add_argument('--num_threads', type=int, default=0, help='number of threads')
  parser.add_argument('--prob_threshold', type=float, default=0.3, help='min probability of detection result')
  args = parser.parse_args()

  interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=args.num_threads)
  interpreter.allocate_tensors()

  labels = load_labels(args.label_file)

  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  cap = cv2.VideoCapture(args.video)
  while(True):
    ret, img = cap.read()
    if ret == False:
      break

    start = time.time()

    data = cv2.resize(img, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    data = np.expand_dims(data, axis=0)
    data = data.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()

    # h: y, w: x
    # ymin, xmin, ymax, xmax
    bbox = interpreter.get_tensor(output_details[0]['index'])
    label = interpreter.get_tensor(output_details[1]['index'])
    prob = interpreter.get_tensor(output_details[2]['index'])

    confidence_filter = prob > 0.3

    output_bboxes = bbox[confidence_filter]
    output_labels = label[confidence_filter].reshape(-1, 1)
    output_probs = prob[confidence_filter].reshape(-1, 1)
    outputs = np.hstack((output_bboxes, output_labels, output_probs))

    elapsed_time = time.time() - start
    print("infer time: ", elapsed_time , "[s]")

    result_img = img.copy()

    for output in outputs:
      x0 = int(output[1] * img.shape[1])
      y0 = int(output[0] * img.shape[0])
      x1 = int(output[3] * img.shape[1])
      y1 = int(output[2] * img.shape[0])
      text = labels[output[4]] + ": " + str(output[5])
      result_img = cv2.putText(result_img, text, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
      result_img = cv2.rectangle(result_img, (x0, y0), (x1, y1), (255, 0, 0))

    cv2.imshow("result", result_img)
    cv2.waitKey(1)
  cv2.destroyAllWindows()
