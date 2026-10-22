from concurrent.futures import ProcessPoolExecutor


def parallel_map(fn, tasks: list, jobs: int) -> list:
    if jobs <= 1:
        return [fn(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, tasks, chunksize=1))
