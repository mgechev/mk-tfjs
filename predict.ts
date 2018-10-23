import * as tf from '@tensorflow/tfjs';
import { loadModel, readInput } from './utils';
require('@tensorflow/tfjs-node');

const loadModels = async () => {
  const mobileNet = await loadModel();
  const model = await tf.loadModel('file://punch_model/model.json');
  return { mobileNet, model };
};

const predict = async (path: string) => {
  const models = await loadModels();
  return Number((models.model.predict(models.mobileNet(readInput(path))) as tf.Tensor1D).dataSync());
};

predict('./hits-aug/358_7.jpg').then(p => console.log(p));
