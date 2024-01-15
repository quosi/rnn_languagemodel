var textData;
var selectedModel;
var model // = tf.loadLayersModel('models/dickens-model_80epochs/model_LSTM.json');
var onTextGenerationCharVar = null;
var generatedTextInput = document.getElementById('generated-text');
const generateLengthInput = document.getElementById('generate-length');
const temperatureInput = document.getElementById('temperature');
const seedTextInput = document.getElementById('seed-text');

function getSelectedModelId() {
    const radioButtons = document.getElementsByName("model");
    for (const radioButton of radioButtons) {
        if (radioButton.checked) {
            return radioButton.value;
        }
    }
}

/**
 * Load chosen (LSTM) model from the given location.
 */
async function loadModel() {
    // Get the selected model value
    console.log(`Loading model...`);
    const modelPath = `models/dickens-model_80epochs/model_LSTM.json`;
    try {
        model = await tf.loadLayersModel(modelPath);
        console.log(`Model loaded from: ${modelPath}`);
        // ToDo : Display network configurations run test data after loading the model
    } catch (error) {
        console.error(`Error loading model: ${error.message}`);
    }
}

/**
 * A function to call each time a character is obtained during text generation.
 * @param {string} char The just-generated character.
 */
async function onTextGenerationChar(char) {
    generatedTextInput.value += char;
    generatedTextInput.scrollTop = generatedTextInput.scrollHeight;
    const charCount = generatedTextInput.value.length;
    const generateLength = parseInt(generateLengthInput.value);
    console.log(`Generating text: ${charCount}/${generateLength} complete.`);
    // allowing the UI to update each time before executing the next iteration
    await tf.nextFrame();
}

/**
 * Draw a sample based on probabilities.
 * @param {tf.Tensor} probs Predicted probability scores, as a 1D `tf.Tensor` of shape `[charSetSize]`.
 * @param {tf.Tensor} temperature Temperature (i.e., a measure of randomness
 *   or diversity) to use during sampling. Number be a number > 0, as a Scalar `tf.Tensor`.
 * @returns {number} The 0-based index for the randomly-drawn sample, in the range of `[0, charSetSize - 1]`.
 */
function sample(probs, temperature) {
  return tf.tidy(() => {
    const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
    const isNormalized = false;
    // `logits` is for a multinomial distribution, scaled by the temperature.
    // We randomly draw a sample from the distribution.
    return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
  });
}

function getFromCharSet(index) {
    const charset = [' ', 'O', 'l', 'i', 'v', 'e', 'r', 'T', 'w', 's', 't', 'o', 'n', 'C', 'h', 'a', 'D', 'c', 'k', 'm', 'b', 'g', 'p', 'E', 'u', 'J', 'G', 'V', '.', 'H', 'B', 'U', 'd', 'Ä', 'Y', ',', 'z', 'f', 'S', 'P', 'W', 'K', 'L', 'B', 'I', 'b', 'l', 'A', 'M', 'F', '7', '1', '8', '2', 'j', 'y', '3', 'R', 'T', 'u', '_', 'n', 'Z', ' ', '(', ')', ':', 'N', '9', '4', '0', '6', ';', '-', 'Q', '5', '!', '–', 'd', '?', 'x', '*', '[', ']', 'q', 'X', 'Y', '>', '<'];
    return charset[index];}

/**
 * Generate text using a next-char-prediction model.
 * @param {tf.Model} model The model object to be used for the text generation,
 *   assumed to have input shape `[null, sampleLen, charSetSize]` and
 *   output shape `[null, charSetSize]`.
 * @param {number[]} sentenceIndices The character indices in the seed sentence.
 * @param {number} length Length of the sentence to generate.
 * @param {number} temperature Temperature value. Must be a number >= 0 and <= 1.
 * @param {(char: string) => Promise<void>} onTextGenerationChar An optional
 *   callback to be invoked each time a character is generated.
 * @returns {string} The generated sentence.
 */
async function generateText(model, textData, sentenceIndices, length, temperature, onTextGenerationCharVar) {

    if (sentenceIndices.length > 0) {

        const sampleLen = 60 // model.inputs[0].shape[1];
        const charSetSize = 89 // model.inputs[0].shape[2];

        // Avoid overwriting the original input.
        sentenceIndices = sentenceIndices.slice();

        let generated = '';
        console.log('LSTM Text wird erstellt...');

        while (generated.length < length) {
            // Encode the current input sequence as a one-hot Tensor.
            const inputBuffer = new tf.TensorBuffer([1, sampleLen, charSetSize]);

            // Make the one-hot encoding of the seeding sentence.
            for (let i = 0; i < sampleLen; ++i) {
                inputBuffer.set(1, 0, i, sentenceIndices[i]);
            }
            const input = inputBuffer.toTensor();

            // Call model.predict() to get the probability values of the next character.
            const output = model.predict(input);
            // Sample randomly based on the probability values.
            const winnerIndex = sample(tf.squeeze(output), temperature);
            const winnerChar = getFromCharSet(winnerIndex);
            if (onTextGenerationCharVar != null) {
                await onTextGenerationChar(winnerChar);
            }

            generated += winnerChar;
            sentenceIndices = sentenceIndices.slice(1);
            sentenceIndices.push(winnerIndex);

            // Memory cleanups.
            input.dispose();
            output.dispose();
        }
    generatedTextInput.value = generated;
    } else {
        generatedTextInput.value = 'Bitte gib ein Word ein.';
    }
    return generatedTextInput.value;
};

async function handleUserInput() {
    // get text base
    const textCorpus = new TextSource('Dickens');
    const text = textCorpus.getText();

    const sampleLen = 60 // model.inputs[0].shape[1];
    const sampleStep = 3; // Step length: how many characters to skip between one example; Default: 3

    if (textData == null){
        textData = new TextData('Dickens', text, sampleLen, sampleStep);
    };
    selectedModel = getSelectedModelId()
    console.log('--> selectedModel: ', selectedModel)
    loadModel()

    if (selectedModel == 'LSTM') {
        const sentenceIndices = [...Array(seedTextInput.value.length).keys()];// Bsp. für eine 10 Buchstaben lange Eingabe: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        const length = generateLengthInput.value;
        const temperature = temperatureInput.value;
        generatedText = generateText(model, textData, sentenceIndices, length, temperature, onTextGenerationCharVar);


    } else {  // ALGO
        console.log('Starting algorithmic prediction ...');
        const userInput = seedTextInput.value.toLowerCase().split(' ');
        const N = 2; // N=2 -> bigram model
        /**
        if (userInput.length !== N - 1) {
            document.getElementById("generated-text").innerHTML = "Bitte gib ein Wort ein.";
        */
        const ngram_freq = create_tokens(text, N);
        const nr = generateLengthInput.value;
        const prediction_nr_words = predict_n_words(userInput, ngram_freq, nr);
        generatedTextInput.value = prediction_nr_words + '.'
        console.log(`Fin ${nr} words prediction: ${prediction_nr_words}`);
    };
};

document.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        handleUserInput();
    };
});
