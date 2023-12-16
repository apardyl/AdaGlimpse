from abc import abstractmethod, ABC
from multiprocessing import get_context
from typing import Tuple, Iterator, List

import torch.utils.data
from torch import Tensor
from torch.multiprocessing import spawn
from torch.multiprocessing.queue import Queue

from architectures.rl.interactive_sampler import InteractiveStatelessSampler
from architectures.rl.shared_memory import SharedMemory


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
    def __init__(self, dataloader: torch.utils.data.DataLoader, max_glimpses: int, glimpse_grid_size: int,
                 batch_size: int, image_size: Tuple[int, int], native_patch_size: Tuple[int, int],
                 device: torch.device, num_parallel_games: int) -> None:
        super().__init__(dataloader, max_glimpses, glimpse_grid_size, batch_size, image_size, native_patch_size, device)
        self.num_parallel_games = num_parallel_games

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

    def __iter__(self) -> Iterator[Tuple[SharedMemory, int]]:
        data_iter = iter(self.dataloader)
        games = []
        cpu_images = []
        for _ in range(self.num_parallel_games):
            games.append(self._build_shared_memory())
            cpu_images.append(torch.zeros(self.batch_size, 3, *self.image_size, device='cpu').share_memory_())
        for game_idx in range(self.num_parallel_games):
            if not self._load_next_batch(data_iter, games[game_idx], cpu_images[game_idx]):
                raise RuntimeError('not enough data to populate glimpse games')

        games_playing = self.num_parallel_games

        mp = get_context('spawn')

        process_queue = mp.Queue()
        ready_queue = mp.Queue()

        context = spawn(fn=glimpse_worker, args=(process_queue, ready_queue, games, cpu_images, self.sampler),
                        nprocs=self.num_parallel_games, join=False, daemon=True)

        for game_idx in range(self.num_parallel_games):
            ready_queue.put(game_idx)

        while games_playing > 0:
            game_idx = ready_queue.get()
            if game_idx is None:
                context.join(2.)
                raise RuntimeError('worker error')

            yield games[game_idx], game_idx

            if not games[game_idx].is_done:
                process_queue.put(game_idx)
            else:
                if self._load_next_batch(data_iter, games[game_idx], cpu_images[game_idx]):
                    ready_queue.put(game_idx)
                else:
                    games_playing -= 1

        # stop workers
        for game_idx in range(self.num_parallel_games):
            process_queue.put(None)
        context.join(timeout=5)
        process_queue.close()
        ready_queue.close()
        for game_idx in range(self.num_parallel_games):
            games[game_idx].close()


def glimpse_engine(*args, num_parallel_games=0, **kwargs):
    if num_parallel_games == 0:
        return SyncGlimpseEngine(*args, **kwargs)
    else:
        return ParallelGlimpseEngine(*args, num_parallel_games=num_parallel_games, **kwargs)
