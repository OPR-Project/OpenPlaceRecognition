"""Estimate memory usage for distance calculation with configurable parameters."""

import argparse
import gc
import time
from typing import Optional

import numpy as np
import psutil
from tqdm import tqdm


def estimate_memory_usage(  # noqa: C901
    num_points: int = 5000,
    dims: int = 3,
    dtype: type = np.float32,
    batch_size: Optional[int] = None,
) -> None:
    """Estimate memory usage for distance calculation with configurable parameters.

    Args:
        num_points: Number of points to use.
        dims: Dimensions of each point.
        dtype: Data type to use for the arrays.
        batch_size: If specified, calculate distances in batches to reduce memory usage.
    """

    def compute_geo_dist_efficient(coords: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Compute pairwise L2 distances over coordinates.

        Args:
            coords: Array of coordinates with shape (n, dims).
            batch_size: If specified, process the computation in batches of this size.
                Reduces memory usage but may increase computation time.

        Returns:
            np.ndarray: Distance matrix with shape (n, n).
        """
        if batch_size is None:
            return np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        else:
            n = coords.shape[0]
            dist_matrix = np.zeros((n, n), dtype=np.float32)
            for i in tqdm(
                range(0, n, batch_size),
                desc="Computing geographic distances",
                leave=False,
            ):
                end_idx = min(i + batch_size, n)
                batch = coords[i:end_idx]
                dist_matrix[i:end_idx] = np.sqrt(
                    ((batch[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2)
                )
            return dist_matrix

    # Get initial memory usage
    process = psutil.Process()
    gc.collect()  # Force garbage collection before measuring
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Create array of target size
    print(f"Creating test array with {num_points} points of {dims} dimensions...")
    coords = np.random.random((num_points, dims)).astype(dtype)

    array_memory = process.memory_info().rss / 1024 / 1024
    print(
        f"Memory after creating array: {array_memory:.2f} MB (added {array_memory - initial_memory:.2f} MB)"
    )

    # Calculate theoretical memory for distance matrix
    distance_matrix_bytes = num_points * num_points * np.dtype(dtype).itemsize
    print(f"Theoretical size of distance matrix: {distance_matrix_bytes / 1024 / 1024:.2f} MB")

    # Print information about the operation
    print(f"Input shape: {coords.shape}, dtype: {coords.dtype}")
    print(f"Expected output shape: ({coords.shape[0]}, {coords.shape[0]})")

    if batch_size:
        print(f"Using batch processing with batch size: {batch_size}")
        batch_memory_estimate = (batch_size * num_points * dims * np.dtype(dtype).itemsize) / 1024 / 1024
        print(f"Estimated peak memory for batch temp arrays: {batch_memory_estimate:.2f} MB")
        print(
            "Potential memory savings: ",
            f"~{(num_points - batch_size) * num_points * dims * np.dtype(dtype).itemsize / 1024 / 1024:.2f} MB",
        )

    # Time the operation and measure peak memory
    print("\nRunning distance calculation...")
    memory_usage = []

    def memory_monitor() -> None:
        """Monitor memory usage and store it in a list."""
        memory_usage.append(process.memory_info().rss / 1024 / 1024)

    start_time = time.time()
    try:
        # Start memory monitoring in a separate thread
        import threading

        stop_monitoring = False

        def monitor_thread() -> None:
            """Thread to monitor memory usage."""
            while not stop_monitoring:
                memory_monitor()
                time.sleep(0.1)

        monitor = threading.Thread(target=monitor_thread)
        monitor.daemon = True
        monitor.start()

        # Run the calculation
        dist_matrix = compute_geo_dist_efficient(coords, batch_size)

        # Stop monitoring
        stop_monitoring = True
        monitor.join()

        calculation_time = time.time() - start_time
        peak_memory = max(memory_usage) if memory_usage else process.memory_info().rss / 1024 / 1024

        print(f"Calculation successful in {calculation_time:.2f} seconds")
        print(f"Output shape: {dist_matrix.shape}, dtype: {dist_matrix.dtype}")
        print(f"Peak memory usage: {peak_memory:.2f} MB (added {peak_memory - array_memory:.2f} MB)")

    except MemoryError:
        print("MemoryError: Not enough memory to complete the operation")
    except Exception as e:
        print(f"Error during calculation: {type(e).__name__}: {e}")

    print("\nMemory cleanup...")
    del coords
    if "dist_matrix" in locals():
        del dist_matrix
    gc.collect()  # Force garbage collection

    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory usage: {final_memory:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test memory usage of distance matrix calculation")
    parser.add_argument("--num-points", type=int, default=5000, help="Number of points to generate")
    parser.add_argument("--dims", type=int, default=3, help="Number of dimensions per point")
    parser.add_argument("--batch-size", type=int, help="If specified, process in batches of this size")
    args = parser.parse_args()

    # Print system memory information
    print("System memory information:")
    virtual_memory = psutil.virtual_memory()
    print(f"Total: {virtual_memory.total / 1024 / 1024:.2f} MB")
    print(f"Available: {virtual_memory.available / 1024 / 1024:.2f} MB")
    print(f"Used: {virtual_memory.used / 1024 / 1024:.2f} MB")
    print(f"Percentage: {virtual_memory.percent}%")
    print()

    estimate_memory_usage(num_points=args.num_points, dims=args.dims, batch_size=args.batch_size)
