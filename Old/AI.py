class AI():
    def __init__(self, name):
        self.name = name
        self.state = "idle"

    def set_state(self, new_state):
        self.state = new_state

    def get_state(self):
        return self.state

    def perform_action(self, action):
        print(f"{self.name} is performing action: {action}")
        self.set_state("busy")