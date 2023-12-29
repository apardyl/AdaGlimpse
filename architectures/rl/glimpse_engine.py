from abc import abstractmethod, ABC
from multiprocessing import get_context
from time import sleep
from typing import Tuple, Iterator, List, Callable, Dict, Iterable

import torch.utils.data
from torch import Tensor
from torch.multiprocessing import spawn
from torch.multiprocessing.queue import Queue

from architectures.rl.interactive_sampler import InteractiveStatelessSampler
from architectures.rl.shared_memory import SharedMemory


class BaseGlimpseEngine(ABC):
    def __init__(self, max_glimpses: int, glimpse_grid_size: int,
                 batch_size: int, image_size: Tuple[int, int], native_patch_size: Tuple[int, int],
                 max_glimpse_size_ratio: float, device: torch.device, create_target_tensor_fn: Callable[[int], Tensor],
                 copy_target_tensor_fn: Callable[[Tensor, Dict[str, Tensor]], None]) -> None:
        self.max_glimpses = max_glimpses
        self.glimpse_grid_size = glimpse_grid_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.native_patch_size = native_patch_size
        self.device = device
        self.create_target_tensor_fn = create_target_tensor_fn
        self.copy_target_tensor_fn = copy_target_tensor_fn

        self.sampler = InteractiveStatelessSampler(
            glimpse_grid_size=self.glimpse_grid_size,
            max_glimpses=self.max_glimpses,
            native_patch_size=self.native_patch_size,
            max_size_ratio=max_glimpse_size_ratio
        )

    def _build_shared_memory(self):
        return SharedMemory(
            max_glimpses=self.max_glimpses, image_size=self.image_size,
            native_patch_size=self.native_patch_size, glimpse_grid_size=self.glimpse_grid_size,
            batch_size=self.batch_size, device=self.device, create_target_tensor_fn=self.create_target_tensor_fn,
            copy_target_tensor_fn=self.copy_target_tensor_fn
        )

    @abstractmethod
    def get_loader(self, dataloader) -> Iterable:
        raise NotImplementedError()


class SyncGlimpseEngine(BaseGlimpseEngine):
    class _GlimpseIterator:
        def __init__(self, engine: 'SyncGlimpseEngine', dataloader):
            self.engine = engine
            self.dataloader = dataloader

        def __iter__(self) -> Iterator[Tuple[SharedMemory, int]]:
            state = self.engine._build_shared_memory()

            for batch in self.dataloader:
                state.set_batch(batch)
                yield state, 0
                while not state.is_done:
                    self.engine.sampler.sample(images_cpu=batch['image'], shared_memory=state)
                    yield state, 0

        def __len__(self) -> int:
            return len(self.dataloader) * (self.engine.max_glimpses + 1)

    def get_loader(self, dataloader) -> Iterable:
        return SyncGlimpseEngine._GlimpseIterator(self, dataloader)


# noinspection PyUnusedLocal
def glimpse_worker(worker_id, request_queue: Queue, response_queue: Queue, games: List[SharedMemory],
                   cpu_images: List[Tensor], sampler: InteractiveStatelessSampler) -> None:
    try:
        while True:
            game_idx = request_queue.get()
            if game_idx is None:
                break  # no more requests

            sampler.sample(images_cpu=cpu_images[game_idx], shared_memory=games[game_idx])
            response_queue.put(game_idx)

        del games

    except Exception:
        response_queue.put(None)
        raise


class ParallelGlimpseEngine(BaseGlimpseEngine):
    def __init__(self, num_parallel_games: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_games = num_parallel_games

        self.games = []
        self.cpu_images = []

        mp = get_context('spawn')

        for _ in range(self.num_parallel_games):
            self.games.append(self._build_shared_memory())
            self.cpu_images.append(torch.zeros(self.batch_size, 3, *self.image_size, device='cpu').share_memory_())

        self.process_queue = mp.Queue()
        self.ready_queue = mp.Queue()

        self.context = spawn(
            fn=glimpse_worker,
            args=(self.process_queue, self.ready_queue, self.games, self.cpu_images, self.sampler),
            nprocs=self.num_parallel_games, join=False, daemon=True
        )

        self.games_playing = 0
        self.version = 0

    def close(self):
        for game_idx in range(self.num_parallel_games):
            self.process_queue.put(None)
        self.context.join(timeout=5)
        self.process_queue.close()
        self.ready_queue.close()
        for game_idx in range(self.num_parallel_games):
            self.games[game_idx].close()

    @staticmethod
    def _load_next_batch(data_iter, game: SharedMemory, cpu_image: Tensor) -> bool:
        try:
            batch = next(data_iter)
        except StopIteration:
            return False
        else:
            game.set_batch(batch)
            cpu_image[:batch['image'].shape[0]].copy_(batch['image'])
            return True

    class _GlimpseIterator:
        def __init__(self, engine: 'ParallelGlimpseEngine', dataloader):
            self.engine = engine
            self.dataloader = dataloader

        def __iter__(self) -> Iterator[Tuple[SharedMemory, int]]:
            if self.engine.games_playing > 0:
                # wait for other games to finish.
                sleep(1)
                # drain worker queue.
                while not self.engine.ready_queue.empty():
                    self.engine.ready_queue.get(timeout=1)

            self.engine.version += 1
            version = self.engine.version

            data_iter = iter(self.dataloader)

            for game_idx in range(self.engine.num_parallel_games):
                if not self.engine._load_next_batch(
                        data_iter, self.engine.games[game_idx], self.engine.cpu_images[game_idx]):
                    raise RuntimeError('not enough data to populate glimpse games')

            self.engine.games_playing = self.engine.num_parallel_games

            for game_idx in range(self.engine.num_parallel_games):
                self.engine.ready_queue.put(game_idx)

            while self.engine.games_playing > 0:
                game_idx = self.engine.ready_queue.get()
                if game_idx is None:
                    self.engine.context.join(2.)
                    raise RuntimeError('worker error')

                yield self.engine.games[game_idx], game_idx
                assert version == self.engine.version  # check if no other process is using the same worker pool

                if not self.engine.games[game_idx].is_done:
                    self.engine.process_queue.put(game_idx)
                else:
                    if self.engine._load_next_batch(data_iter, self.engine.games[game_idx],
                                                    self.engine.cpu_images[game_idx]):
                        self.engine.ready_queue.put(game_idx)
                    else:
                        self.engine.games_playing -= 1

        def __len__(self) -> int:
            return len(self.dataloader) * (self.engine.max_glimpses + 1)

    def get_loader(self, dataloader) -> Iterable:
        return ParallelGlimpseEngine._GlimpseIterator(self, dataloader)


def glimpse_engine(*args, num_parallel_games=0, **kwargs):
    if num_parallel_games == 0:
        return SyncGlimpseEngine(*args, **kwargs)
    else:
        return ParallelGlimpseEngine(*args, num_parallel_games=num_parallel_games, **kwargs)
