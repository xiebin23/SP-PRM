class BaseScorer:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_batched_reward(self, *args, **kwargs):
        raise NotImplementedError
