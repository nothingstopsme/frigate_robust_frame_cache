import contextlib
import multiprocessing as mp
import os
import queue
import signal
import subprocess as sp
import threading
import time
import unittest
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np

from frigate.camera import CameraMetrics
from frigate.config.camera import CameraConfig
from frigate.util.image import SharedMemoryFrameManager
from frigate.video import CameraCaptureRunner


@unittest.skipIf(
    not hasattr(SharedMemoryFrameManager, "HEADER_NBYTES"),
    "test not compatible with the current version of SharedMemoryFrameManager",
)
class TestSafeReadWrite(unittest.TestCase):
    def test_serial_verification(self) -> None:
        frame_manager = SharedMemoryFrameManager()
        size = 8
        shm_name = "test_camera"
        frame_manager.create(shm_name, size)

        unique_name0 = f"({shm_name},00000000)"
        unique_name1 = f"({shm_name},01000000)"
        prn_generator = np.random.default_rng(seed=55667788)
        frame_data0 = np.ndarray(size, dtype=np.uint8, buffer=prn_generator.bytes(size))
        frame_data1 = np.where(frame_data0 == 255, 0, frame_data0 + 1)

        # Test 1: no frame has been written, so nothing returned
        with frame_manager.read(unique_name0) as frame_handle:
            self.assertIsNone(frame_handle)

        # Test 2: the first frame is written into the shm block named with the value of shm_name
        # => The serial number starts at 0
        # => This frame is now associated with a unique name equal to the value of unique_name0
        with frame_manager.write(shm_name) as frame_handle:
            self.assertIsNotNone(frame_handle)
            self.assertEqual(frame_handle.name_comps.to_str(), unique_name0)
            self.assertEqual(frame_handle.buffer.nbytes, size)
            frame_handle.buffer[...] = frame_data0

        # Test 3: calling read() with the value of unique_name0 should
        # return the frame written at Test 2
        with frame_manager.read(unique_name0) as frame_handle:
            self.assertIsNotNone(frame_handle)
            self.assertEqual(frame_handle.name_comps.shm_name, shm_name)
            self.assertIsNotNone(frame_handle.buffer)
            self.assertEqual(frame_handle.buffer.shape, frame_data0.shape)
            self.assertTrue((frame_handle.buffer == frame_data0).all())

        # Test 4: calling read() with the value of unique_name1 should
        # not return the frame named with the value of unique_name0
        with frame_manager.read(unique_name1) as frame_handle:
            self.assertIsNone(frame_handle)

        # Test 5: the second frame is written into the shm block named with the value of shm_name
        # => The serial number is advanced to 1
        # => This frame is now associated with a unique name equal to the value of unique_name1
        with frame_manager.write(shm_name) as frame_handle:
            self.assertIsNotNone(frame_handle)
            self.assertEqual(frame_handle.name_comps.to_str(), unique_name1)
            self.assertEqual(frame_handle.buffer.nbytes, size)
            frame_handle.buffer[...] = frame_data1

        # Test 6: calling read() with the value of unique_name0 should
        # not return the frame named with the value of unique_name1
        with frame_manager.read(unique_name0) as frame_handle:
            self.assertIsNone(frame_handle)

        # Test 7: calling read() with the value of unique_name1 should
        # return the frame written at Test 5
        with frame_manager.read(unique_name1) as frame_handle:
            self.assertIsNotNone(frame_handle)
            self.assertEqual(frame_handle.name_comps.shm_name, shm_name)
            self.assertIsNotNone(frame_handle.buffer)
            self.assertEqual(frame_handle.buffer.shape, frame_data1.shape)
            self.assertTrue((frame_handle.buffer == frame_data1).all())


class TestFrameCache(unittest.TestCase):
    def __get_sigint_handler(self, child_proc):
        def handler(self, signum, frame):
            if signum == signal.SIGINT:
                child_proc.send_signal(signum)
                raise KeyboardInterrupt()

        return handler

    def __launch_test(
        self,
        frame_shape: tuple[int, int],
        producer_runs: int,
        test_func: Callable[[mp.Queue, mp.Event], None],
    ):

        if producer_runs > 256 or producer_runs < 1:
            raise ValueError(
                "Invalid parameters; please make sure 1 <= producer_runs <= 256"
            )

        config = MagicMock(CameraConfig)
        config.name = "TestCamera"
        config.enabled = True

        sync_manager = mp.Manager()
        camera_metrics = CameraMetrics(sync_manager)

        frame_manager = SharedMemoryFrameManager()
        stop_event = sync_manager.Event()

        # Note:
        # 1. Unfortunately, there seem no public ways for accessing "maxsize"
        # designated during construction of a Queue instance, so a test is conducted to check its value
        #
        # 2. The minimum shm_frame_count recommended in camera/maintainer.py is 20, which is
        # much larger than the max size of frame_queue (currently fixed internally at 2);
        # here shm_frame_count is simply set to twice as large as frame_queue,
        # so that the frame producer can quickly overwhelm the shared memory cache

        # maxsize test
        while True:
            try:
                camera_metrics.frame_queue.put((None, -1), False)
            except queue.Full:
                break

        max_frame_queue_size = camera_metrics.frame_queue.qsize()

        # Resetting the queue by recreating it
        camera_metrics.frame_queue = sync_manager.Queue(max_frame_queue_size)

        shm_frame_count = max_frame_queue_size * 2

        frame_size = int(np.prod(frame_shape))
        frame_rate = 5.0
        start_delay = 1.0
        mock_producer_cmd = (
            "import time, sys, struct\n"
            f"time.sleep({start_delay})\n"
            f"for run in range({producer_runs}):\n"
            f"    for frame_index in range({shm_frame_count}):\n"
            f"        sys.stdout.buffer.write(struct.pack('=B', run)*{frame_size})\n"
            f"        time.sleep({1 / frame_rate})"
        )

        try:
            for i in range(shm_frame_count):
                frame_manager.create(f"{config.name}_frame{i}", frame_size)

            existing_handler = signal.getsignal(signal.SIGINT)
            # Turning off the signal handler for SIGINT before starting
            # a child process, to make sure that that the following SIGINTs being handled
            # must go to our custom handler once that child process is created
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            producer = sp.Popen(
                ["python3", "-u", "-c", mock_producer_cmd],
                stdout=sp.PIPE,
                stdin=sp.PIPE,
                bufsize=frame_size * 10,
                start_new_session=True,
            )
            signal.signal(signal.SIGINT, self.__get_sigint_handler(producer))

            with patch("frigate.config.camera.updater.CameraConfigUpdateSubscriber"):

                def start_capturing():
                    runner = CameraCaptureRunner(
                        config,
                        shm_frame_count,
                        0,
                        producer,
                        frame_shape,
                        camera_metrics.frame_queue,
                        camera_metrics.camera_fps,
                        camera_metrics.skipped_fps,
                        stop_event,
                    )
                    runner.start()

                capturer = mp.Process(target=start_capturing)
                capturer.start()

            test_exeception = None

            def run_test():
                """run_test() is run in a test thread"""
                nonlocal test_exeception, camera_metrics, stop_event
                try:
                    test_func(camera_metrics.frame_queue, stop_event)
                except Exception as e:
                    # Keeping the reference to the exception so that we can raise it later
                    # in the main thread, and then raising it again as if it is not catched
                    test_exeception = e
                    raise e

            test_thread = threading.Thread(target=run_test)
            test_thread.start()

            # Waiting for the producer to finish
            while True:
                try:
                    print(".", end="")
                    producer.wait(3)
                    break
                except sp.TimeoutExpired:
                    pass

            print()

            # Reinstalling the handler used for SIGINT before this test after the producer is finished
            signal.signal(signal.SIGINT, existing_handler)

            # Raising stop_event
            stop_event.set()

            capturer.join()
            test_thread.join()
            if test_exeception:
                raise test_exeception

        finally:
            for i in range(shm_frame_count):
                frame_manager.delete(f"{config.name}_frame{i}")

    def __get_frame_reader(
        self, frame_manager: SharedMemoryFrameManager
    ) -> Callable[[str, Sequence[int]], contextlib.AbstractContextManager[Any]]:
        if hasattr(frame_manager, "HEADER_NBYTES"):

            def read_frame(
                frame_name: str, frame_shape: Sequence[int]
            ) -> contextlib.AbstractContextManager[
                SharedMemoryFrameManager.FrameHandle
            ]:
                return frame_manager.read(frame_name, frame_shape)
        else:
            # In this case, testing might be performed against an older version of SharedMemoryFrameManager
            def read_frame(
                frame_name: str, frame_shape: Sequence[int]
            ) -> contextlib.AbstractContextManager[SimpleNamespace]:
                return contextlib.nullcontext(
                    SimpleNamespace(buffer=frame_manager.get(frame_name, frame_shape))
                )

        return read_frame

    def __get_frame_writer(
        self, frame_manager: SharedMemoryFrameManager
    ) -> Callable[[str, Sequence[int]], contextlib.AbstractContextManager[Any]]:
        if hasattr(frame_manager, "HEADER_NBYTES"):

            def write_frame(
                frame_name: str, frame_shape: Sequence[int]
            ) -> contextlib.AbstractContextManager[
                SharedMemoryFrameManager.FrameHandle
            ]:
                return frame_manager.write(frame_name, frame_shape)
        else:
            # In this case, testing might be performed against an older version of SharedMemoryFrameManager
            def write_frame(
                frame_name: str, frame_shape: Sequence[int]
            ) -> contextlib.AbstractContextManager[SimpleNamespace]:
                return contextlib.nullcontext(
                    SimpleNamespace(
                        buffer=np.ndarray(
                            frame_shape,
                            dtype=np.uint8,
                            buffer=frame_manager.write(frame_name),
                        )
                    )
                )

        return write_frame

    def test_lasting_frame_readonly_use(self):
        """This test simulates the case where there are processes working on already obtained frame for a extended period of time,
        and to check if the data integrity of those frame is maintained under such a condition.
        """
        use_table: dict[str, tuple[mp.Process, mp.Value]] = {}
        frame_shape = (256, 256)

        def test_impl(frame_queue: mp.Queue, stop_event: mp.Event):
            """For each frame recieved, repeatedly performing a readonly operation which
            simply checks if every entry has the same value
            (frames generated by the mock producer are filled with its run index at the moment of generation)
            """

            def use_frame_readonly(
                target_frame_name: str, corrupt_frame_count: mp.Value
            ):
                frame_manager = SharedMemoryFrameManager()
                read_frame = self.__get_frame_reader(frame_manager)

                with read_frame(target_frame_name, frame_shape) as frame_handle:
                    self.assertIsNotNone(frame_handle)
                    frame = frame_handle.buffer
                    while not stop_event.is_set():
                        # Note: since other processes will wait until this process
                        # has finished before read corrupt_frame_count
                        # it is fine to conduct read-modify operations in this non-atomic fashion
                        corrupt_frame_count.value += int((frame[0, 0] != frame).any())

            while not stop_event.is_set():
                try:
                    frame_name = frame_queue.get(False)[0]
                except queue.Empty:
                    continue

                if frame_name not in use_table:
                    corrupt_frame_count = mp.Value("I", 0)
                    frame_user = mp.Process(
                        target=use_frame_readonly,
                        args=(frame_name, corrupt_frame_count),
                    )
                    frame_user.start()
                    use_table[frame_name] = (frame_user, corrupt_frame_count)

        # Since frame corruption only happens by chance, more producer_runs, despite being a rather extreme condition,
        # are used to make it more likely to emerge
        self.__launch_test(
            frame_shape=frame_shape, producer_runs=10, test_func=test_impl
        )

        corrupt_frame_count = 0
        for frame_user, corrupt_frame_count_per_block in use_table.values():
            frame_user.join()
            corrupt_frame_count += corrupt_frame_count_per_block.value

        self.assertEqual(
            corrupt_frame_count, 0, f"corrupt frame count = {corrupt_frame_count}"
        )

    def test_frame_consumption_delay(self):
        """This test simulates the case where the frame queue turns full due to a delay in frame comsuption,
        and to check if the data integrity of currently queued frames is maintained under such a condition.
        """

        frame_shape = (256, 256)
        overwritten_frame_count = 0

        def test_impl(frame_queue: mp.Queue, stop_event: mp.Event):
            """Reading from frame_queue only starts after stop_event is raised,
            to simulate a delay in frame consumption
            """
            nonlocal overwritten_frame_count

            frame_manager = SharedMemoryFrameManager()
            read_frame = self.__get_frame_reader(frame_manager)

            while not stop_event.is_set():
                time.sleep(1)

            while True:
                try:
                    frame_name = frame_queue.get(False)[0]
                except queue.Empty:
                    break

                with read_frame(frame_name, frame_shape) as frame_handle:
                    self.assertIsNotNone(frame_handle)
                    frame = frame_handle.buffer
                    overwritten_frame_count += int((frame != 0).any())

        # by setting producer_runs to 2, the mock producer will generate a sufficient number of frames
        # (twice as many as the total shared memory blocks) to allow us easily to stuff the frame queue
        # full
        self.__launch_test(
            frame_shape=frame_shape, producer_runs=2, test_func=test_impl
        )

        self.assertEqual(
            overwritten_frame_count,
            0,
            f"overwritten frame count = {overwritten_frame_count}",
        )

    def __skip_if_benchmark_off(self):
        if os.environ.get("Benchmark_" + type(self).__name__, "off").lower() in {
            "off",
            "0",
        }:
            self.skipTest(
                f"Benchmarking from {__name__}.{type(self).__name__} is skipped"
            )

    def test_read_write_speed(self) -> None:
        """This test measures average time taken to complete a full fream read/write operation"""
        self.__skip_if_benchmark_off()

        warmup_iterations = 5
        measurements = [None] * 40
        num_of_excluded_outliers = 10

        def summarize(type_name):
            measurements.sort()
            avg = (
                sum(measurements[num_of_excluded_outliers:-num_of_excluded_outliers])
                / (len(measurements) - num_of_excluded_outliers * 2)
                * 1000.0
            )
            print(f"\nAverage {type_name} time = {avg} ms", end="\n\n")

        frame_manager = SharedMemoryFrameManager()
        shape = (320, 320)
        size = np.prod(shape)
        shm_name = "test_camera"
        frame_manager.create(shm_name, size)

        prn_generator = np.random.default_rng(seed=55667788)
        frame_data = [
            np.ndarray(shape, dtype=np.uint8, buffer=prn_generator.bytes(size))
        ]
        frame_data.append(np.where(frame_data[-1] == 255, 0, frame_data[-1] + 1))
        frame_data_index = 0

        write_frame_fn = self.__get_frame_writer(frame_manager)
        read_frame_fn = self.__get_frame_reader(frame_manager)

        # Putting initial data into the created frame slot
        with write_frame_fn(shm_name, shape) as frame_handle:
            frame_handle.buffer[...] = frame_data[frame_data_index]

        for measurement_index in range(warmup_iterations + len(measurements)):
            start_time = time.perf_counter()
            with read_frame_fn(shm_name, shape) as frame_handle:
                # Doing something that actually accesses every data entry
                unused_copy = np.copy(frame_handle.buffer)

            duration = time.perf_counter() - start_time
            if measurement_index >= warmup_iterations:
                measurements[measurement_index - warmup_iterations] = duration

        summarize("read")

        for measurement_index in range(warmup_iterations + len(measurements)):
            frame_data_index = (frame_data_index + 1) & 1
            start_time = time.perf_counter()
            with write_frame_fn(shm_name, shape) as frame_handle:
                # Doing something that actually accesses every data entry
                frame_handle.buffer[...] = frame_data[frame_data_index]

            duration = time.perf_counter() - start_time
            if measurement_index >= warmup_iterations:
                measurements[measurement_index - warmup_iterations] = duration

        summarize("write")

        frame_manager.cleanup()

    def test_read_speed_under_contention(self):
        """This test measures the average duration between 2 successful frame reads
        in a mock setup involving multiple frame readers and a single frame writer
        """

        self.__skip_if_benchmark_off()

        frame_shape = (320, 320)
        measurements = [None] * 40
        num_of_consumers = 10
        producer_runs_per_measurement = 5
        num_of_excluded_outliers = 10

        def consume_frame(
            msg_queue: mp.Queue, avg_frame_waiting_time: mp.Value, stop_event: mp.Event
        ):
            frame_manager = SharedMemoryFrameManager()
            read_frame = self.__get_frame_reader(frame_manager)

            frames_received = 0
            avg_time = 0
            start_time = time.perf_counter()
            end_time = None
            while not stop_event.is_set():
                try:
                    frame_name = msg_queue.get(False)
                except queue.Empty:
                    continue

                with read_frame(frame_name, frame_shape) as frame_handle:
                    if frame_handle:
                        end_time = time.perf_counter()
                        # Doing something that actually accesses every data entry to simulate frame data use
                        unused_copy = np.copy(frame_handle.buffer)

                if end_time:
                    frames_received += 1
                    avg_time += ((end_time - start_time) - avg_time) / frames_received
                    start_time = time.perf_counter()
                    end_time = None

            avg_frame_waiting_time.value = avg_time

        FrameConsumer = NamedTuple(
            "FrameConsumer",
            (
                ("process", mp.Process),
                ("msg_queue", mp.Queue),
                ("avg_frame_waiting_time", mp.Value),
            ),
        )

        def create_test_func(measurement_index):
            consumers = []
            consumer_stop_event = mp.Event()
            # Creating and Starting all consumer processs in advance
            for _ in range(num_of_consumers):
                msg_queue = mp.Queue()
                avg_frame_waiting_time = mp.Value("d", 0)
                consumers.append(
                    FrameConsumer(
                        mp.Process(
                            target=consume_frame,
                            args=(
                                msg_queue,
                                avg_frame_waiting_time,
                                consumer_stop_event,
                            ),
                        ),
                        msg_queue,
                        avg_frame_waiting_time,
                    )
                )
                consumers[-1].process.start()

            def test_impl(frame_queue: mp.Queue, stop_event: mp.Event):
                """This function simply relays frame names received from frame_queue to all the frame consumers"""

                while not stop_event.is_set():
                    try:
                        frame_name = frame_queue.get(False)[0]
                    except queue.Empty:
                        continue

                    for msg_queue in map(lambda c: c.msg_queue, consumers):
                        try:
                            msg_queue.put(frame_name)
                        except queue.Full:
                            pass

                # stop_event has been raised, so we can signal consumers to stop
                consumer_stop_event.set()
                avg_frame_waiting_time = 0
                for process, _, avg_frame_waiting_time_per_consumer in consumers:
                    process.join()
                    avg_frame_waiting_time += avg_frame_waiting_time_per_consumer.value

                measurements[measurement_index] = avg_frame_waiting_time / len(
                    consumers
                )

            return test_impl

        for measurement_index in range(len(measurements)):
            print(f"\r\x1b[0KMeasurement {measurement_index}", end="")

            self.__launch_test(
                frame_shape=frame_shape,
                producer_runs=producer_runs_per_measurement,
                test_func=create_test_func(measurement_index),
            )

        measurements.sort()
        avg = (
            sum(measurements[num_of_excluded_outliers:-num_of_excluded_outliers])
            / (len(measurements) - num_of_excluded_outliers * 2)
            * 1000.0
        )
        print(f"\nAverage duration between 2 successful frame reads = {avg} ms")


if __name__ == "__main__":
    unittest.main()
