class CoreModelNotTrained(Exception):
    def __init__(self, message="No saved instance of the core model was found. "
                               "Please make sure the model is trained and saved."):
        super().__init__(message)


class SpeechRecognizerNotTrained(Exception):
    def __init__(self, message="No saved instance of the speech recognizer model was found. "
                               "Please make sure the model is trained and saved."):
        super().__init__(message)


class LanguageModelNotTrained(Exception):
    def __init__(self, message="No saved instance of the language model was found. "
                               "Please make sure the model is trained and saved."):
        super().__init__(message)


class SpeechSynthesizerCannotBeLoaded(Exception):
    def __init__(self, message="No saved instance of the synthesizer model was found. "
                               "Please make sure there is a saved instance of the model"):
        super().__init__(message)
