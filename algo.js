function read_file(identifier) {
    textObject = new TextSource(identifier);
    const corpus = textObject.getText();
    return corpus;
}

function create_tokens(text, n = 2) {

    var cleanText = '';
    // Filter out non-alphanumeric words
    for (var i=0; i < text.length; i++) {
        cleanText += text.charAt(i).replace(/[«?!*#~,;&._»<>"'()-]/g, ' ');
    };
    const words = cleanText.split(/\s+/);

    // create ngram object with ngram as key and occurrence in corpus as value.
    var ngram_dict = {};

    for (let i = 0; i < words.length-1; i++) {
        var key = [ words[i], words[i+1] ];
        if (key in ngram_dict) {
            ngram_dict[key] += 1;
        } else {
            ngram_dict[key] = 1;
        };
    };
    // NOTE : ngrams_list should be list of tuples of word pairs of the whole text
    // [('the', 'project'), ('project', 'gutenberg'), ('gutenberg', 'ebook'), ... ] 56629 items for Balzac
    // NOTE : Use tuples as key for the ngram_dictionary and store occurrences of tuples as vale
    // {('the', 'project'): 31, ('project', 'gutenberg'): 30, ('gutenberg', 'ebook'): 4, ... } 41536 items for Balzac
    return ngram_dict;
}

function predict_next_word(prefix, ngram_freq) {
    const matchingNgrams = Object.entries(ngram_freq).filter(([key]) => key.startsWith(prefix.slice(-1)));
    if (matchingNgrams.length === 0) {
        return "Das Wort kenne ich nicht";
    }

    let randomIndex =  Math.max(0, Math.floor(Math.random() * matchingNgrams.length));
    var res = matchingNgrams[randomIndex][0].split(",")[1];
    return res
}

function predict_n_words(prefix, ngram_freq, n = 1) {

    for (let i = 0; i < n; i++) {
        const predWord = predict_next_word(prefix.slice(-1), ngram_freq);

        if (predWord === "Das Wort kenne ich nicht") {
            return "Das Wort kenne ich nicht";
        }
        prefix.push(predWord);
    }
    const allWords = prefix.join(' ');
    const capitalizedWords = allWords.charAt(0).toUpperCase() + allWords.slice(1);
    return capitalizedWords
}

