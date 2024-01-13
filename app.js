var model;
var textData;
var onTextGenerationCharVar = null;
var generatedTextInput = document.getElementById('generated-text');
const generateLengthInput = document.getElementById('generate-length');
const temperatureInput = document.getElementById('temperature');
const seedTextInput = document.getElementById('seed-text');
// const outputDiv = document.getElementById('output');

/**
 * Load chosen (LSTM) model from the given location.
 */
async function loadModel() {
    // Get the selected model value
    console.log(`Loading model...`);
    const selectedModel = document.getElementById("model").value;
    const modelPath = `models/dickens-model_80epochs/model_${selectedModel}.json`;
    try {
        model = await tf.loadLayersModel(modelPath);
        console.log(`Model loaded from: ${modelPath}`);
        // Display network configurations run test data after loading the model
        // ...
    } catch (error) {
        console.error(`Error loading model: ${error.message}`);
    }
}


/**
 * A function to call each time a character is obtained during text generation.
 * @param {string} char The just-generated character.
 */
async function onTextGenerationChar(char) {
    // ToDo : below is set to '' (empty) during generateText, right?
    generatedTextInput.value += char;
    generatedTextInput.scrollTop = generatedTextInput.scrollHeight;
    const charCount = generatedTextInput.value.length;
    const generateLength = parseInt(generateLengthInput.value);
    console.log(`Generating text: ${charCount}/${generateLength} complete.`);
    // allowing the UI to update each time before executing the next iteration
    await tf.nextFrame();
}


/**
 * Generate text using a next-char-prediction model.
 * @param {tf.Model} model The model object to be used for the text generation,
 *   assumed to have input shape `[null, sampleLen, charSetSize]` and
 *   output shape `[null, charSetSize]`.
 * @param {number[]} sentenceIndices The character indices in the seed sentence.
 * @param {number} length Length of the sentence to generate.
 * @param {number} temperature Temperature value. Must be a number >= 0 and
 *   <= 1.
 * @param {(char: string) => Promise<void>} onTextGenerationChar An optional
 *   callback to be invoked each time a character is generated.
 * @returns {string} The generated sentence.
 */
async function generateText(model, textData, sentenceIndices, length, temperature, onTextGenerationCharVar) {

    if (sentenceIndices.length > 0) {

        const sampleLen = model.inputs[0].shape[1];
        const charSetSize = model.inputs[0].shape[2];

        // Avoid overwriting the original input.
        sentenceIndices = sentenceIndices.slice();

        generatedTextInput.value = '';
        // ToDo : use above oder below line ?? or both ??
        let generated = '';
        console.out('Text wird erstellt...');

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
            const winnerIndex = sample(tf.squeeze(output), temperature.value);
            const winnerChar = textData.getFromCharSet(winnerIndex);
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
    return generated;
    } else {
        generatedTextInput.textContent = 'Bitte gib ein Word ein.';
    }
};

async function handleUserInput() {
    // get text base
    const localTextDataPath = './data/Gutenberg_eBook_de_oliver_twist.txt';
    const text = fs.readFileSync(localTextDataPath, {encoding: 'utf-8', flag: 'r'});
    console.log(text.slice(0,30));
    const sampleLen = model.inputs[0].shape[1];
    const sampleStep = 3; // Step length: how many characters to skip between one example; Default: 3
    if (textData == null){
        textDate = new TextData('Dickens', text, sampleLen, sampleStep);
    };

    const sentenceIndices = [...Array(seedTextInput.length).keys()];// Bsp. für eine 10 Buchstaben lange Eingabe: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const length = generateLengthInput.value;
    const temperature = temperatureInput.value;
    // const onTextGenerationCharVar = null;
    generatedText = generateText(model, textData, sentenceIndices, length, temperature, onTextGenerationCharVar);
    document.getElementById('generatedTextInput').innerHTML = generatedText; // .innerText is also used sometimes
}

//DONE : build event listener for each character typed in the seed-text-input field,
//DONE : perform this, after DOM was fully loaded to prevent error
document.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        handleUserInput();
    };
});
