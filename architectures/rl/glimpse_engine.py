from abc import abstractmethod, ABC
from multiprocessing import get_context
from typing import Tuple, Iterator, List, Dict

import torch.utils.data
from torch import Tensor
from torch.multiprocessing import spawn
from torch.multiprocessing.queue import Queue

from architectures.rl.interactive_sampler import InteractiveStatelessSampler
from architectures.rl.shared_memory import SharedMemory
from datasets import ImageNet1k


class BaseGlimpseEngine(ABC):
    def __init__(self, dataloader: torch.utils.data.DataLoader, max_glimpses: int, glimpse_grid_size: int,
                 batch_size: int, image_size: Tuple[int, int], native_patch_size: Tuple[int, int],
                 device: torch.device) -> None:
        self.dataloader = dataloader
        self.max_glimpses = max_glimpses
        self.glimpse_grid_size = glimpse_grid_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.native_patch_size = native_patch_size
        self.device = device

        self.sampler = InteractiveStatelessSampler(
            glimpse_grid_size=self.glimpse_grid_size,
            max_glimpses=self.max_glimpses,
            native_patch_size=self.native_patch_size
        )

    def _build_shared_memory(self):
        return SharedMemory(
            max_glimpses=self.max_glimpses, image_size=self.image_size,
            native_patch_size=self.native_patch_size, glimpse_grid_size=self.glimpse_grid_size,
            batch_size=self.batch_size, device=self.device
        )

    def __len__(self) -> int:
        return len(self.dataloader) * (self.max_glimpses + 1)

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[SharedMemory, int]]:
        raise NotImplementedError()


class SyncGlimpseEngine(BaseGlimpseEngine):
    def __iter__(self) -> Iterator[Tuple[SharedMemory, int]]:
        state = self._build_shared_memory()

        for batch in self.dataloader:
            state.set_batch(batch)
            yield state, 0
            while not state.is_done:
                self.sampler.sample(images_cpu=batch['image'], shared_memory=state)
                yield state, 0


def glimpse_worker(worker_id, request_queue: Queue, response_queue: Queue, games: List[SharedMemory],
                   sampler: InteractiveStatelessSampler) -> None:
    images = None

    try:

        while True:
            request = request_queue.get()
            if request is None:
                return
            game_idx, new_game = request

            if new_game:
                images = games[game_idx].images.cpu()
            else:
                assert images is not None
                sampler.sample(images_cpu=images, shared_memory=games[game_idx])

            response_queue.put(game_idx)

    except Exception:
        response_queue.put(None)
        raise


class ParallelGlimpseEngine(BaseGlimpseEngine):
    def __init__(self, dataloader: torch.utils.data.DataLoader, max_glimpses: int, glimpse_grid_size: int,
                 batch_size: int, image_size: Tuple[int, int], native_patch_size: Tuple[int, int],
                 device: torch.device, num_parallel_games: int) -> None:
        super().__init__(dataloader, max_glimpses, glimpse_grid_size, batch_size, image_size, native_patch_size, device)
        self.num_parallel_games = num_parallel_games

    @staticmethod
    def _load_next_batch(data_iter, game: SharedMemory) -> bool:
        try:
            batch = next(data_iter)
        except StopIteration:
            return False
        else:
            game.set_batch(batch)
            return True

    def __iter__(self) -> Iterator[Tuple[SharedMemory, int]]:
        data_iter = iter(self.dataloader)
        games = []
        for _ in range(self.num_parallel_games):
            games.append(self._build_shared_memory())
        for game_idx in range(self.num_parallel_games):
            if not self._load_next_batch(data_iter, games[game_idx]):
                raise RuntimeError('not enough data to populate glimpse games')

        games_playing = self.num_parallel_games

        mp = get_context('spawn')

        request_queue = mp.Queue()
        response_queue = mp.Queue()

        context = spawn(fn=glimpse_worker, args=(request_queue, response_queue, games, self.sampler),
                        nprocs=self.num_parallel_games,
                        join=False)

        for game_idx in range(self.num_parallel_games):
            request_queue.put((game_idx, True))

        while games_playing > 0:
            game_idx = response_queue.get()
            if game_idx is None:
                context.join(2.)
                raise RuntimeError('worker error')

            yield games[game_idx], game_idx

            if not games[game_idx].is_done:
                request_queue.put((game_idx, False))
            else:
                if self._load_next_batch(data_iter, games[game_idx]):
                    request_queue.put((game_idx, True))
                else:
                    request_queue.put(None)
                    games_playing -= 1

        context.join(timeout=2.)


def test():
    data = ImageNet1k(data_dir='/home/adam/datasets/imagenet', train_batch_size=2, eval_batch_size=2,
                      num_workers=0, always_drop_last=True)
    data.setup('validate')
    dataloader = data.val_dataloader()

    engine = ParallelGlimpseEngine(dataloader=dataloader, max_glimpses=3, glimpse_grid_size=2,
                                   native_patch_size=(16, 16),
                                   batch_size=4, device=torch.device('cuda'), image_size=(224, 224),
                                   num_parallel_games=1)

    for state in engine:
        print(state)


if __name__ == '__main__':
    test()
