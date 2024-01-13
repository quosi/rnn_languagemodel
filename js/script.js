// Define global variables
let model;
const OOV_TOKEN = '<OOV>';

// Function to preprocess and tokenize the text data

function preprocessTextData(textData, sequenceLength) {
    // Step 1: Clean text
    const cleanedText = cleanText(textData);

    // Step 2: Tokenize text
    const tokens = tokenizeText(cleanedText, null); // wordIndex is null for now

    // Step 3: Create sequences
    const sequences = createSequences(tokens, sequenceLength);

    // Step 4: Create word index
    const wordIndex = createWordIndex(tokens);

    // Step 5: Convert to numerical representation
    const numericSequences = convertToNumeric(sequences, wordIndex);

    return { numericSequences, wordIndex };
}

function cleanText(text) {
    // Remove non-alphanumeric characters
    text = text.replace(/[^a-zA-Z0-9\s]/g, '');
    // Convert to lowercase
    return text.toLowerCase();
}

// Function to tokenize text incl. OOV handling
function tokenizeText(text, wordIndex) {
    return text.split(/\s+/).map(word => wordIndex[word] || OOV_TOKEN);
}

function createSequences(tokens, sequenceLength) {
    const sequences = [];
    for (let i = 0; i < tokens.length - sequenceLength; i++) {
        const inputSequence = tokens.slice(i, i + sequenceLength);
        const targetWord = tokens[i + sequenceLength];
        sequences.push({ input: inputSequence, target: targetWord });
    }
    return sequences;
}

function createWordIndex(tokens) {
    const uniqueWords = [...new Set(tokens)];
    const wordIndex = {};

    // Include OOV token
    wordIndex[OOV_TOKEN] = 0;

    uniqueWords.forEach((word, index) => {
        wordIndex[word] = index + 1; // Start indexing from 1
    });
    return wordIndex;
}

function convertToNumeric(sequences, wordIndex) {
    return sequences.map(seq => ({
        input: seq.input.map(word => wordIndex[word]),
        target: wordIndex[seq.target],
    }));
}

// Function to prepare training data
function prepareTrainingData(sequences) {
    // Initialize arrays to store input (X) and output (y) data
    const X = [];
    const y = [];

    // Iterate through each sequence
    sequences.forEach(sequence => {
        const inputSequence = sequence.input;
        const targetWord = sequence.target;

        // Add the input sequence to X
        X.push(inputSequence);

        // Add the target word to y
        y.push(targetWord);
    });

    // Convert arrays to TensorFlow tensors
    const XTensor = tf.tensor2d(X);
    const yTensor = tf.oneHot(y, Object.keys(wordIndex).length); // Use the size of the word index

    return { X: XTensor, y: yTensor };
}


// Function to preprocess input sequence during inference handling OOV
function preprocessInputSequence(inputText, sequenceLength, wordIndex) {
    const cleanedText = cleanText(inputText);
    const tokens = tokenizeText(cleanedText, wordIndex);

    // Handle OOV words by replacing them with the OOV token
    const inputSequence = tokens.map(word => wordIndex[word] || OOV_TOKEN);

    // Ensure the sequence has the desired length (pad or truncate if necessary)
    while (inputSequence.length < sequenceLength) {
        inputSequence.unshift(wordIndex[OOV_TOKEN]);  // Pad from the beginning with OOV token
    }

    return inputSequence.slice(0, sequenceLength);
}

// Function to train the model, trigger on button click
async function trainModel() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;

    if (files.length > 0) {
        const sequenceLength = 10;  // Adjust as needed
        const allTextData = [];

        // Read and concatenate content of all selected files
        for (const file of files) {
            const textData = await file.text();
            allTextData.push(textData);
        }

        // Concatenate text data from all files
        const combinedTextData = allTextData.join('');

        // Preprocess text data including wordIndex
        const { numericSequences, wordIndex } = preprocessTextData(combinedTextData, sequenceLength);

        // Prepare training data
        const trainingData = prepareTrainingData(numericSequences);
        const { X, y } = trainingData;

        // Build and compile the model
        model = tf.sequential();

        // Train the model
        await model.fit(X, y, {
            epochs: 50,
            batchSize: 64,
            callbacks: tf.callbacks.earlyStopping({ monitor: 'loss', patience: 3 }),
        });

        alert('Training completed!');
    } else {
        alert('Please select one or more files.');
    }
}

//  inference code for prediction
// const inputSequence = preprocessInputSequence('input_text', sequenceLength, wordIndex);
// const prediction = model.predict(inputSequence);
// ...

// Additional functions...
// cleanText, createSequences, createWordIndex, convertToNumeric
// ...