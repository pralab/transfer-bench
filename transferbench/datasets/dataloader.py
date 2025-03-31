r"""DataLoader with device support."""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.utils.data import (
    DataLoader as DataLoader_,
)
from torch.utils.data import (
    Dataset,
    Sampler,
    default_collate,
)


class DataLoader(DataLoader_):
    r"""Extension of torch.utils.data.DataLoader to handle device."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[list] | Iterable[list] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list], Any] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable[[int], None] | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        device: None | torch.device = None,
    ) -> None:
        """."""
        if collate_fn is None:
            collate_fn = default_collate

        if device is not None:

            def collate_fn(x: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
                return tuple(x_.to(device) for x_ in default_collate(x))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
