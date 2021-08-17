<br />
<p align="center">
  <a href="https://github.com/naderabdalghani/project-rev">
    <img src="assets/logo.jpg" alt="Logo" width="208" height="76">
  </a>

  <h3 align="center">project rev</h3>

  <p align="center">
    A proof-of-concept audio-interactive personalized chatbot based on Ted Mosby, a character from the renowned TV show "How I Met Your Mother"
  </p>
</p>

## Table of Contents

- [About the Project](#about-the-project)
  - [Idea](#idea)
  - [Disclaimer](#disclaimer)
  - [Project Block Diagram](#project-block-diagram)
  - [Demo](#demo)
  - [Speech Recognizer](#speech-recognizer)
  - [Language Model](#language-model)
  - [Core Module](#core-module)
  - [Generator](#generator)
  - [Speech Synthesizer](#speech-synthesizer)
  - [Possible Improvements](#possible-improvements)
  - [Built With](#built-with)
  - [Datasets](#datasets)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running](#running)
- [Acknowledgements](#acknowledgements)

## About The Project

### Idea

This project is about creating a chatbot that simulates a certain persona, whether a real one or a virtual one, through an audio-interactive interface where users talk to it using their voice and consequently, the bot responds using a voice that resembles the simulated person's voice to an extent.

### Disclaimer

Some of the modules represented below are implemented for educational purposes only and despite them having near state-of-the-art implementations (e.g speech recognizer model inspired by [DeepSpeech 2](https://arxiv.org/abs/1512.02595) architecture), they weren't trained or implemented to give the best possible results due to resources restrictions; hence it's best advised that they are substituted with much better ones if the goal is top-notch results and configurability. For instance, the generator module, despite being fully functional and generating good results, it doesn't have the luxuries that [Hugging Face's `model.generate()`](https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate) sports such as repetition, length and diversity penalties.

### Project Block Diagram

![Project Architecture][project-architecture]

A simple graphical interface is used to wrap the modules illustrated above into a single interface with which the user can interact.

### Demo

<a href="https://drive.google.com/file/d/1sCZlh2Mdv3WStFJyo87aRjUIYbiCZkfz/view" target="_blank" rel="noopener noreferrer">
  <img src="assets/demo-thumbnail.jpg" alt="project rev demo" width="500">
</a>

### Speech Recognizer

A module powered by deep neural networks and signal analysis and processing techniques to convert users' audio signals to text.

#### **Flow Architecture**

![Speech Recognizer][speech-recognizer]

#### **Evaluation**

After training the model using Mel Spectrogram features on [LibriSpeech ASR corpus](https://www.openslr.org/12) training set of 360 hours "clean" speech for 40 epochs split into 2 15-hour sessions on a Nvidia V100 instance, the model yielded a `word error rate (WER) = 0.2097601` and a `character error rate (CER) = 0.06480708`

#### **Example Runs**

- Run #1:

  - Input: "He hoped there would be _**stew**_ for dinner, turnips and _**carrots**_ and bruised potatoes and fat mutton pieces to be _**ladled**_ out in thick _**peppered**_ _**flour**_, _**fat**_ _**and**_ sauce"

  - Prediction: "he hoped there would be _**sto**_ for dinner turnips and _**carats**_ and bruised potatoes and fat mutton pieces to be _**laitled**_ out in thick _**pepperd**_ _**flowr**_ _**fattaind**_ sauce"

- Run #2:

  - Input: "Also, a _**popular contrivance**_ _**whereby**_ love making _**may be**_ suspended but not stopped during the picnic _**season**_"

  - Prediction: "also a _**popularcandrivans**_ _**wher by**_ love making _**maybe**_ suspended but not stopped during the picnic _**xeason**_"

### Language Model

A purely statistical n-grams model trained on [Tweets Blogs News](https://www.kaggle.com/crmercado/tweets-blogs-news-swiftkey-dataset-4million) is used to filter semantic and syntactic errors resulted unintentionally from the previous stage, the speech recognition phase.

#### **Flow Architecture**

![Language Model][language-model]

#### **Evaluation**

Using perplexity as a performance metric, the module resulted in a `perplexity = 6.96` on a 2-million-sentence dataset

#### **Example Runs**

- Run #1:

  - Input: "_**hellow**_ ted how is it _**goink**_"

  - Output: "_**hello**_ ted how is it _**going**_"

- Run #2:

  - Input: "i _**red**_ a book"

  - Output: "i _**read**_ a book"

- Run #3:

  - Input: "_**thee**_ _**logik**_ of hogan _**winnink**_ the _**wordl**_ _**titls**_ at the end _**mmade**_ no sense"

  - Output: "_**the**_ _**logic**_ of hogan _**winnin**_ the _**woudl**_ _**title**_ at the end _**made**_ no sense"

### Core Module

A transformer-based module that takes the user's latest utterance as an input and outputs the logits (scores) of each word in the vocabulary to be the succeeding word of the given sequence.

#### **Dataset**

The dataset used to fine-tune a pre-trained [BlenderBot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/) transformer model instance is basically the script of the first 6 seasons or so of "How I Met Your Mother". It's scraped off [this website](https://transcripts.foreverdreaming.org/viewforum.php?f=177) and processed to hold to the format shown below using [this script](core/scripts/dataset_downloader.py).

Dataset format:

```xml
<bot> Kids, I'm going to tell you an incredible story. The story of how I met your mother
<s> Are we being punished for something?
<bot> No
<s> Yeah, is this going to take a while?
<bot> Yes. Twenty-five years ago, before I was dad, I had this whole other life.
<bot> It was way back in 2005. I was twenty-seven just starting to make it as an architect and living in New York with my friend Marshall, my best...
```

The `<bot>` tag marks the sentences being said by the character that the core module should predict their responses, on the other hand, the `<s>` tag marks the sentences being said by characters other than the character to be simulated. This, in fact, gives the flexibility of training the core module on other characters or even using real life conversations of a particular person, but this is only possible if the conversations data is given in the specified format above.

#### **Example Runs**

- Run #1:

  - User: "What do you think of Stella"

  - Model: "I love her so much. She's the best thing that has ever happened to me. In a nutshell, I think I'm in love with her."

- Run #2:

  - User: "tell me more about the Olive theory"

  - Model: "The olive theory is based on my friends Marshal and Lilly. He hates olives, she loves them"

- Run #3:

  - User: "are we playing laser tag tomorrow night?"

  - Model: "Yeah, sure. Laser tag, it's a good idea. We'll do it."

### Generator

This module is solely in charge of generating the predicted response of the simulated persona based on the logits (scores) given from the core module.

It is configurable to use different decoding methods like:

- Greedy search
- Top-K sampling
- Top-p (nucleus) sampling
- Best search (experimental implementation, not practical to use at all)
- Beam search

### Speech Synthesizer

[misbah4064/Real-Time-Voice-Cloning](https://github.com/misbah4064/Real-Time-Voice-Cloning) is used to convert core module output text to speech with a similar voice to the simulated person.

### Possible Improvements

- Use a larger dataset for the speech recognizer (e.g [Mozilla's Common Voice](https://commonvoice.mozilla.org/en/datasets))
- Train a bigger instance of the speech recognizer model for many more epochs
- Improve speech recognizer capability to recognize voices not present in the dataset
- Use a bigger corpus such as [OpenSubtitles' OPUS corpus](https://opus.nlpl.eu/OpenSubtitles2018.php) for training the n-gram language model or, better yet, use a neural network architecture instead of a probabilistic model for improved results.
- Clean and extend core module dataset to the whole 9 seasons
- Use a better and bigger pre-trained model for core module fine-tuning (e.g [BlenderBot 2.0](https://ai.facebook.com/blog/blender-bot-2-an-open-source-chatbot-that-builds-long-term-memory-and-searches-the-internet/))
- Solve core module occasional factual incorrectness by incorporating some kind of a knowledge base or a long-term memory with the transformer-based model

### Built With

- [Python 3.7](https://www.python.org/downloads/release/python-370/)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Google Cloud Services](https://cloud.google.com)
- [Google Colab](https://research.google.com)
- [Microsoft Azure](https://azure.microsoft.com)
- [Hugging Face's Transformers](https://huggingface.co/transformers/index.html)
- [Facebook AI's BlenderBot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/)
- [Comet](https://www.comet.ml)
- [PyTorch](https://pytorch.org/)
- [Ray Tune](https://docs.ray.io/en/ray-0.4.0/tune.html)
- [Flask](https://flask.palletsprojects.com/en/2.0.x/)
- [misbah4064/Real-Time-Voice-Cloning](https://github.com/misbah4064/Real-Time-Voice-Cloning)

### Datasets

- [LibriSpeech ASR corpus](https://www.openslr.org/12)
- [How I Met Your Mother Script](https://transcripts.foreverdreaming.org/viewforum.php?f=177)
- [Tweets Blogs News](https://www.kaggle.com/crmercado/tweets-blogs-news-swiftkey-dataset-4million)

## Getting Started

### Prerequisites

- Setup Python using this [link](https://realpython.com/installing-python/)
- Download and install [FFmpeg](https://ffmpeg.org/download.html)
- Install [`requirements.txt`](requirements.txt) packages using the following line to skip errors should a package fail to install:

  `cat requirements.txt | xargs -n 1 pip install`

- Create a `keys.py` file in the project directory with the following content:

  ```py
  COMET_API_KEY = "zJemHQ8mJtC2Cgv6bxUcsBxxd"
  FLASK_SECRET_KEY = "8b3HefzzLm2qYEce#"
  ```

- To use Comet while training, set `COMET_API_KEY` with a valid API key which can be obtained free of charge from [here](https://www.comet.ml/signup)

- Create a `models` directory in the project directory. This directory should include saved trained instances or dictionaries needed to run the project. You can either train each module and it would automatically save the required files in `/models` or you can easily download those files from [here](https://www.mediafire.com/folder/qrbs04tf1ge6z/project_rev).

- `/models` directory structure should be similar to the following for successful runs:

  ```
  models
  ├── core-model-3160-1.8512854990749796
  │   ├── config.json
  │   └── pytorch_model.bin
  │
  ├── speech-synthesizer
  │   ├── synthesizer
  │   │   ├── checkpoint
  │   │   ├── tacotron_model.ckpt-278000.data-00000-of-00001
  │   │   ├── tacotron_model.ckpt-278000.index
  │   │   └── tacotron_model.ckpt-278000.meta
  │   ├── encoder.pt
  │   ├── TED_VOICE_SAMPLE.wav
  │   └── vocoder.pt
  │
  ├── bigrams_tuples
  ├── names
  ├── speech-recognizer-mel-spectrogram.pt
  ├── trigrams_tuples
  └── unigrams_tuples
  ```

### Running

- Each module can be configured using its own `config.py` file. However, the Flask web app which serves as the 'wrapping' module for the project is configured through [`app_config.py`](app_config.py)

- Perhaps the most interesting parameter in [`app_config.py`](app_config.py) is the `APP_MODE` parameter. It has 3 possible values:

  - `APP_MODE = "TEXT_CHAT_MODE"` would render a simple text chat interface that essentially serves as a modular test for the core module.

    ![Text Chat Interface][text-chat-interface]

  - `APP_MODE = "VOICE_CHAT_MODE"` would render a voice chat interface where all modules are loaded and used for inference. **WARNING:** this mode require a relatively powerful machine with at least 16 GB of memory, so please run with caution.

    ![Voice Chat Interface][voice-chat-interface]

  - Due to the former mode being resource-intensive and the results of the speech recognizer is dependant on the loaded saved instance, a lighter mode is implemented. `APP_MODE = "VOICE_CHAT_LITE_MODE"` is identical to the previous mode in regards to the interface, however, it skips loading both the speech recognizer module and the language model and instead, uses [Web API's SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition).

- Finally, run the following line and launch the Flask app by going to [`http://127.0.0.1:5000/`](http://127.0.0.1:5000/) using a web browser:

  ```
  python app.py
  ```

## Acknowledgements

- [DeepLearning.AI — Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing)
- [Udacity — Natural Language Processing Nanodegree](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892)
- [The A.I. Hacker — Build an AI Voice Assistant with PyTorch](https://www.youtube.com/playlist?list=PL5rWfvZIL-NpFXM9nFr15RmEEh4F4ePZW)
- [Thomas Wolf — How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)
- [Nathan Cooper — How to fine-tune the DialoGPT model on a new dataset for open-dialog conversational chatbots](https://colab.research.google.com/github/ncoop57/i-am-a-nerd/blob/master/_notebooks/2020-05-12-chatbot-part-1.ipynb)
- [Patrick von Platen — How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
- [Chiara Campagnola — Perplexity in Language Models](https://towardsdatascience.com/perplexity-in-language-models-87a196019a94)
- [Roller et al. — Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637)
- [Michael Nguyen — Building an end-to-end Speech Recognition model in PyTorch](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch)
- [Daniel Jurafsky and James H. Martin's Speech and Language Processing — Chapter 3 | N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

[project-architecture]: assets/project-architecture.png
[speech-recognizer]: assets/speech-recognizer.png
[language-model]: assets/language-model.png
[text-chat-interface]: assets/text-chat-interface.png
[voice-chat-interface]: assets/voice-chat-interface.png
