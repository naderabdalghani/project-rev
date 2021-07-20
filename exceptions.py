class CoreModelNotTrained(Exception):
    def __init__(self, message="No saved instance of the core model was found. "
                               "Please make sure the model is trained and saved."):
        super().__init__(message)
