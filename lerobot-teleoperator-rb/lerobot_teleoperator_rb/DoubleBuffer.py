"""Shared-memory double buffer used by the MQ3 receiver process."""

from multiprocessing import shared_memory

import numpy as np


class DoubleBuffer:
    def __init__(
        self,
        dtype: np.dtype,
        buffer_count: int = 2,
        create: bool = True,
        shm_name: str = "double_buffer_shm",
    ):
        self.dtype = dtype
        self.create = create

        self.header_dtype = np.dtype(
            [
                ("current_index", np.int32),
            ]
        )

        self.buffer_count = buffer_count

        self.total_dtype = np.dtype(
            [
                ("header", self.header_dtype),
                ("buffers", dtype, (self.buffer_count,)),
            ]
        )

        size = self.total_dtype.itemsize

        self.shm = shared_memory.SharedMemory(
            create=create,
            size=size,
            name=shm_name,
        )

        self.memory = np.ndarray(
            shape=(),
            dtype=self.total_dtype,
            buffer=self.shm.buf,
        )
        self.prev_index = -1

        if create:
            self.memory["header"]["current_index"] = 0

    @property
    def name(self):
        return self.shm.name

    def write(self, data):
        """Write a structured NumPy element into the hidden buffer."""

        current_index = self.memory["header"]["current_index"]

        # Write away from the buffer currently visible to the reader.
        write_index = (current_index + 1) % self.buffer_count
        self.memory["buffers"][write_index] = data

        # Publish only after the complete value has been written.
        self.memory["header"]["current_index"] = write_index

    def read(self):
        """Return a copy of the currently published structured element."""

        index = self.memory["header"]["current_index"]
        return self.memory["buffers"][index].copy()

    def close(self):
        self.shm.close()
        if self.create:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
