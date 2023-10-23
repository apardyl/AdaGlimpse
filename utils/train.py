import os
import time


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def create_checkpoint_dir(base_checkpoint_dir):
    """
    Creates an arbitrary checkpoint dir. Either with a provided name or just with a date, i.e.
    checkpoints/test_ --> checkpoints/test_<date>
    checkpoints/ --> checkpoints/<date>
    Returns: checkpoint dir path

    """
    current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
    checkpoint_dir = base_checkpoint_dir + current_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
