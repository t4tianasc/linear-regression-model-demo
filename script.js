var stopTraining;
const visorInstance = tfvis.visor();
if (!visorInstance.isOpen()) {
  visorInstance.toggle();
}

async function getData() {
  const housesDataRaw = await fetch("https://static.platzi.com/media/public/uploads/datos-entrenamiento_15cd99ce-3561-494e-8f56-9492d4e86438.json");
  const housesData = await housesDataRaw.json();

  var cleanedData = housesData.map(house => ({
    price: house.Precio,
    rooms: house.NumeroDeCuartosPromedio
  }))
  cleanedData = cleanedData.filter(house => (
    house.price != null && house.rooms != null
  ))
  return cleanedData;
}

async function seeInferenceCurve() {
  var data = await getData();
  var tensorData = await convertDataToTensors(data);
  const {inputsMax, inputsMin, tagsMin, tagsMax} = tensorData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0,1,100);
    const preds = modelo.predict(xs.reshape([100,1]));

    const denormX = xs
      .mul(inputsMax.sub(inputsMin))
      .add(inputsMin);

    const denormY = preds
      .mul(tagsMax.sub(tagsMin))
      .add(tagsMin);
    
    return [denormX.dataSync(), denormY.dataSync()];
  });

  const predictionPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = data.map(d => ({
    x: d.rooms, y: d.price,
  }));

  tfvis.render.scatterplot(
    {name: 'Predictions vs Originals'},
    {values: [originalPoints, predictionPoints], series: ['originals', 'predictions']},
    {
      xLabel: 'Rooms',
      yLabel: 'Price',
      height: 300
    }
  );
}
async function loadModel() {
  const uploadJSONInput = document.getElementById('upload-json');
  const uploadWeightsInput = document.getElementById('upload-weights');
  modelo = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
  console.log("Model loaded");
}

function visualizeData(data){
  const mappedValues = data.map(d => ({
    x: d.rooms,
    y: d.price
  }));
  tfvis.render.scatterplot(
    {name: 'Rooms vs Price'},
    {values: mappedValues},
    {
      xLabel: 'Rooms',
      yLabel: 'Price',
      height: 300
    }
  );
}

function createModel() {
  const new_model = tf.sequential();
  new_model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  new_model.add(tf.layers.dense({units: 1, useBias: true}));
  return new_model;
}

const optimizer = tf.train.adam();
const lost_function = tf.losses.meanSquaredError;
const metric = ['mse'];

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: optimizer,
    loss: lost_function,
    metrics: metric
  })

  const surface = {name: 'show.history live', tab: 'Training'};
  const batchSize = 28;
  const epochs = 50;
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
  const saveResult = await modelo.save('downloads://regresion-model');
}

async function showData() {
  var data = await getData();
  visualizeData(data);
}

function convertDataToTensors(data){
  return tf.tidy(() => {
    tf.util.shuffle(data);
    const inputs = data.map(d => d.rooms);
    const tags = data.map(d => d.price);
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

async function startTraining(){
  var data = await getData();
  modelo = createModel();
  const tensorData = convertDataToTensors(data);
  const {inputs, tags} = tensorData;
  trainModel(modelo, inputs, tags);
}

showData();