var stopTraining;
const visorInstance = tfvis.visor();
var dataXlabel;
var dataYlabel;
var data;

function openVisor() {
  if (!visorInstance.isOpen()) {
    visorInstance.toggle();
  }
}

const optimizer = tf.train.adam();
const lost_function = tf.losses.meanSquaredError;
const metric = ['mse'];

async function getData() {
  try {
    const UrlJsonData = document.getElementById('url-json-data').value;
    dataXlabel = document.getElementById('data-x-variable').value;
    dataYlabel = document.getElementById('data-y-variable').value;
    const objectsDataRaw = await fetch(UrlJsonData);
    const objectsData = await objectsDataRaw.json();

    var cleanedData = objectsData.map(object => ({
      dataY: parseFloat(object[dataYlabel]),
      dataX: parseFloat(object[dataXlabel])
    }))
    cleanedData = cleanedData.filter(object => (
      object.dataY != null && object.dataX != null
    ))
    return cleanedData;
  } catch (error) {
    showNotification("error", "Make sure the input data from the URL is correct");
  }
}

async function seeInferenceCurve() {
  try {
    var tensorData = await convertDataToTensors(data);
    const { inputsMax, inputsMin, tagsMin, tagsMax } = tensorData;

    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100);
      const preds = modelo.predict(xs.reshape([100, 1]));

      const denormX = xs
        .mul(inputsMax.sub(inputsMin))
        .add(inputsMin);

      const denormY = preds
        .mul(tagsMax.sub(tagsMin))
        .add(tagsMin);

      return [denormX.dataSync(), denormY.dataSync()];
    });

    const predictionPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] }
    });

    const originalPoints = data.map(d => ({
      x: d.dataX, y: d.dataY,
    }));

    tfvis.render.scatterplot(
      { name: 'Predictions vs Originals' },
      { values: [originalPoints, predictionPoints], series: ['originals', 'predictions'] },
      {
        xLabel: 'Rooms',
        yLabel: 'Price',
        height: 300,
        zoomToFit: true
      }
    );
  } catch (error) {
    showNotification("error", "Start training a new model or load one");
  }
}

function showNotification(type, msg) {
  var notification = document.getElementById("notification");
  notification.innerHTML = msg;

  notification.classList.remove('notification-success', 'notification-error');
  if (type === 'success') {
    notification.classList.add('notification-success');
  } else if (type === 'error') {
    notification.classList.add('notification-error');
  }

  notification.style.opacity = 1;
  setTimeout(function () {
    notification.style.opacity = 0;
  }, 3000);
}

async function loadModel() {
  const uploadJSONInput = document.getElementById('upload-json');
  const uploadWeightsInput = document.getElementById('upload-weights');
  if (uploadJSONInput.value && uploadWeightsInput.value) {
    modelo = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
    showNotification("success", "Model loaded!");
  } else {
    showNotification("error", "Upload the files");
  }
}

function visualizeData(data) {
  const mappedValues = data.map(d => ({
    x: d.dataX,
    y: d.dataY
  }));
  tfvis.render.scatterplot(
    { name: dataXlabel + ' vs ' + dataYlabel },
    { values: mappedValues },
    {
      xLabel: dataXlabel,
      yLabel: dataYlabel,
      height: 300,
      zoomToFit: true
    }
  );
}

function createModel() {
  const new_model = tf.sequential();
  new_model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  new_model.add(tf.layers.dense({ units: 1, useBias: true }));
  return new_model;
}

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: optimizer,
    loss: lost_function,
    metrics: metric
  })

  const surface = { name: 'show.history live', tab: 'Training' };
  const batchSize = 28;
  const epochs = 100;
  const history = [];

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        history.push(log);
        tfvis.show.history(surface, history, ['loss', 'mse']);
        if (stopTraining) {
          modelo.stopTraining = true;
        }
      }
    }
  })
}

async function saveModel() {
  try {
    const saveResult = await modelo.save('downloads://regresion-model');
  } catch (error) {
    showNotification("error", "Click on 'Start training' to create a new model.");
  }
}

async function showData() {
  data = await getData();
  visualizeData(data);
}

function convertDataToTensors(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);
    const inputs = data.map(d => d.dataX);
    const tags = data.map(d => d.dataY);
    const tensorInputs = tf.tensor2d(inputs, [inputs.length, 1]);
    const tensorTags = tf.tensor2d(tags, [tags.length, 1]);

    const inputsMax = tensorInputs.max();
    const inputsMin = tensorInputs.min();
    const tagsMax = tensorTags.max();
    const tagsMin = tensorTags.min();

    // Normalized inputs: math operation (inputData-min)/(max-min)
    const normalizedInputs = tensorInputs.sub(inputsMin).div(inputsMax.sub(inputsMin));
    const normalizedTags = tensorTags.sub(tagsMin).div(tagsMax.sub(tagsMin));

    return {
      inputs: normalizedInputs,
      tags: normalizedTags,
      inputsMax,
      inputsMin,
      tagsMax,
      tagsMin
    }
  });
}

async function startTraining() {
  modelo = createModel();
  const tensorData = convertDataToTensors(data);
  const { inputs, tags } = tensorData;
  trainModel(modelo, inputs, tags);
}
