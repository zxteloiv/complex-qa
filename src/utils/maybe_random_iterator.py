from trialbot.data.iterators import RandomIterator

class MaybeRandomIterator(RandomIterator):
    def __next__(self):
        data = super().__next__()
        while data is None:
            self.logger.warning('empty data is returned by the translator, try the next instance automatically')
            data = super().__next__()
            if self.is_new_epoch:
                raise StopIteration
        return data
