from trialbot.training import Updater


class DummyUpdater(Updater):
    def __init__(self):
        """
        An updater that does nothing but immediately stops the loop of the current epochs.
        """
        super().__init__()

    def update_epoch(self):
        self.stop_epoch()
        raise StopIteration
