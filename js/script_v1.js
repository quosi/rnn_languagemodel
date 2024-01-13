const chatLog = document.getElementById('chat-log');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Load the pre-trained model
async function loadModel() {
    const model = await tf.loadLayersModel('model/model.json');
    return model;
}

// Generate a response from the model
async function generateResponse(model, message) {
    const inputTensor = tf.tensor2d([message.map(char => char.charCodeAt(0))]);
    const outputTensor = model.predict(inputTensor);
    const outputArray = Array.from(outputTensor.dataSync()[0]);
    const response = outputArray.map(code => String.fromCharCode(code)).join('');
    return response;
}

// Add message to the chat log
function addMessageToChatLog(message, isUser) {
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message');
    if (isUser) {
        messageContainer.classList.add('user');
    } else {
        messageContainer.classList.add('bot');
    }
    messageContainer.textContent = message;
    chatLog.appendChild(messageContainer);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// Handle user input and generate response
async function handleUserInput() {
    const userMessage = userInput.value;
    userInput.value = '';

    addMessageToChatLog(userMessage, true);

    const response = await generateResponse(model, userMessage);
    addMessageToChatLog(response, false);
}

// Load the model and set up event listeners
loadModel().then(loadedModel => {
    sendBtn.addEventListener('click', handleUserInput);
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            handleUserInput();
        }
    });
});
