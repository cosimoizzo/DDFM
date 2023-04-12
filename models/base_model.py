class BaseModel:
    """
    An empty base class for the models.
    """

    def __int__(self, *args, **kwargs):
        super().__int__()

    def fit(self, **kwargs):
        """
        Fit method.
        """
        pass

    def predict(self, **kwargs):
        """
        Predict method
        """
        pass
