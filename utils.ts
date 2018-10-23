import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as jpeg from 'jpeg-js';
const mobilenet = require('@tensorflow-models/mobilenet');

const TotalChannels = 3;

const readImage = path => {
  const buf = fs.readFileSync(path);
  const pixels = jpeg.decode(buf, true);
  return pixels;
};

const imageByteArray = (image, numChannels) => {
  const pixels = image.data;
  const numPixels = image.width * image.height;
  const values = new Int32Array(numPixels * numChannels);

  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numChannels; ++channel) {
      values[i * numChannels + channel] = pixels[i * 4 + channel];
    }
  }

  return values;
};

export const readInput = img => imageToInput(readImage(img), TotalChannels);

const imageToInput = (image, numChannels) => {
  const values = imageByteArray(image, numChannels);
  const outShape = [image.height, image.width, numChannels] as [number, number, number];
  const input = tf.tensor3d(values, outShape, 'int32');

  return input;
};

const Layer = 'global_average_pooling2d_1';
const ModelPath = './model/model.json';
export const loadModel = async () => {
  const mn = new mobilenet.MobileNet(1, 1);
  mn.path = `file://${ModelPath}`;
  await mn.load();
  return (input): tf.Tensor1D => mn.infer(input, Layer).reshape([1024]);
};
