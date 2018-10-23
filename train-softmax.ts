import { loadModel, readInput } from './utils';

import * as tf from '@tensorflow/tfjs';
require('@tensorflow/tfjs-node');

const Hits = 'hits-aug';
const Kicks = 'kicks-aug';
const Negative = 'no-hits-aug';
const Epochs = 500;
const BatchSize = 0.1;
const InputShape = 1024;

const train = async () => {
  const mobileNet = await loadModel();
  const model = tf.sequential();
  model.add(tf.layers.inputLayer({ inputShape: [InputShape] }));
  model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  await model.compile({
    optimizer: tf.train.adam(1e-6),
    loss: tf.losses.sigmoidCrossEntropy,
    metrics: ['accuracy']
  });

  const hits = require('fs')
    .readdirSync(Hits)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Hits}/${f}`);

  const kicks = require('fs')
    .readdirSync(Kicks)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Kicks}/${f}`);

  const negatives = require('fs')
    .readdirSync(Negative)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Negative}/${f}`);

  console.log('Building the training set');

  const ys = tf.tensor2d(
    new Array(hits.length)
      .fill([1, 0, 0])
      .concat(new Array(kicks.length).fill([0, 1, 0]))
      .concat(new Array(negatives.length).fill([0, 0, 1])),
    [hits.length + kicks.length + negatives.length, 3]
  );

  console.log('Getting the punches');
  const xs: tf.Tensor2D = tf.stack(
    hits
      .map((path: string) => mobileNet(readInput(path)))
      .concat(kicks.map((path: string) => mobileNet(readInput(path))))
      .concat(negatives.map((path: string) => mobileNet(readInput(path))))
  ) as tf.Tensor2D;

  console.log('Fitting the model');
  await model.fit(xs, ys, {
    epochs: Epochs,
    batchSize: parseInt(((hits.length + kicks.length + negatives.length) * BatchSize).toFixed(0)),
    callbacks: {
      onBatchEnd: async (_, logs) => {
        console.log('Cost: %s, accuracy: %s', logs.loss.toFixed(5), logs.acc.toFixed(5));
        await tf.nextFrame();
      }
    }
  });

  console.log('Saving the model');
  await model.save('file://punch_kick_simplified');
};

train();
