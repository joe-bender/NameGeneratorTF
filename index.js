// Constants
const sequenceLength = 15;
const lowercaseLetters = 'abcdefghijklmnopqrstuvwxyz'
const chars = lowercaseLetters + ' ';

async function loadModel() {
  let model = await tf.loadModel('models/js/model.json');
  return model;
}

function genName(model, firstLetter) {
  let name = firstLetter;
  for (let i_seq = 0; i_seq < sequenceLength; i_seq++) {
    let x = nameToX(name);
    let y_pred = model.predict(x);
    let nameInts = y_pred.argMax(2).squeeze().dataSync();
    let char = iToChar(nameInts[i_seq]);
    name += char;
  }
  // capitalize name
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function charToI(char) {
  return chars.indexOf(char);
}

function iToChar(i) {
  return chars[i];
}

function nameToX(name) {
  let namePadded = name.padEnd(sequenceLength);
  let nameInts = [];
  for (let char of namePadded) {
    nameInts.push(charToI(char));
  }
  nameTensor = tf.tensor2d([nameInts])
  return nameTensor;
}

loadModel().then(function(model) {
  for (let char of lowercaseLetters) {
    let name = genName(model, char);
    console.log(name);
  }
});
