def tokenlize_text(max_num_words, max_seq_length, x_train):
    
    from keras_preprocessing.sequence import pad_sequences
    from keras_preprocessing.text import Tokenizer
    print("tokenlizing texts...")
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(x_train)
    sequences = tokenizer.texts_to_sequences(x_train)
    word_index = tokenizer.word_index
    x_train = pad_sequences(sequences, maxlen=max_seq_length)
    print("data readed and convert to %d length sequences" % max_seq_length)
    return x_train, word_index