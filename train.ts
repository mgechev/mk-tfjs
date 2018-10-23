import { loadModel, readInput } from './utils';

import * as tf from '@tensorflow/tfjs';
require('@tensorflow/tfjs-node');

const Hits = 'hits-aug';
const Negative = 'no-hits-aug';
const Epochs = 50;
const BatchSize = 0.1;

const train = async () => {
  const mobileNet = await loadModel();
  const model = tf.sequential();
  model.add(tf.layers.inputLayer({ inputShape: [1024] }));
  model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  await model.compile({
    optimizer: tf.train.adam(0.00001),
    loss: tf.losses.sigmoidCrossEntropy,
    metrics: ['accuracy']
  });

  const hits = require('fs')
    .readdirSync(Hits)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Hits}/${f}`);

  const negatives = require('fs')
    .readdirSync(Negative)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Negative}/${f}`);

  console.log('Building the training set');

  const ys = tf.tensor1d(new Array(hits.length).fill(1).concat(new Array(negatives.length).fill(0)));

  console.log('Getting the punches');
  const xs: tf.Tensor2D = tf.stack(
    hits
      .map((path: string) => mobileNet(readInput(path)))
      .concat(negatives.map((path: string) => mobileNet(readInput(path))))
  ) as tf.Tensor2D;
  await model.fit(xs, ys, {
    epochs: Epochs,
    batchSize: parseInt(((hits.length + negatives.length) * BatchSize).toFixed(0)),
    callbacks: {
      onBatchEnd: async (_, logs) => {
        console.log('Cost: %s, accuracy: %s', logs.loss.toFixed(5), logs.acc.toFixed(5));
        await tf.nextFrame();
      }
    }
  });

  await model.save('file://punch_simplified');
};

train();
