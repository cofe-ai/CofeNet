import torch
from torch.utils.data import DataLoader, _utils


class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class _MYBaseDataLoaderIter(object):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.dataset_kind = loader.dataset_kind
        self.auto_collation = loader._auto_collation
        self.drop_last = loader.drop_last
        self.index_sampler = loader._index_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.collate_fn = loader.collate_fn
        self.sampler_iter = iter(self.index_sampler)

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self.sampler_iter)  # may raise StopIteration

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.index_sampler)

    def __getstate__(self):
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _MYSingleProcessDataLoaderIter(_MYBaseDataLoaderIter):
    def __init__(self, loader):
        super(_MYSingleProcessDataLoaderIter, self).__init__(loader)
        assert self.timeout == 0
        assert self.num_workers == 0

        self.dataset_fetcher = _DatasetKind.create_fetcher(
            self.dataset_kind, self.dataset, self.auto_collation, self.collate_fn, self.drop_last)

    def __next__(self):
        index = self._next_index()  # may raise StopIteration
        data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
        if self.pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data

    next = __next__  # Python 2 compatibility


class SingleDataLoader(DataLoader):
    def __init__(self, dataset, **kargs):
        assert kargs.get('num_workers', 0) == 0
        super(SingleDataLoader, self).__init__(dataset, **kargs)

    def __iter__(self):
        return _MYSingleProcessDataLoaderIter(self)
