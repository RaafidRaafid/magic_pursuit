class staticEnv:

    @staticmethod
    def next_state(statee, actions):
        raise NotImplementedError

    @staticmethod
    def is_done_state(state, step_idx):
        raise NotImplementedError

    @staticmethod
    def initial_state():
        raise NotImplementedError

    @staticmethod
    def get_return(state, agentType, idx, root_state=None):
        raise NotImplementedError
