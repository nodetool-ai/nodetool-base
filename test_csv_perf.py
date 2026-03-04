import asyncio
import csv
import io
import time
import tempfile
import os
import aiofiles

data = [{"a": i, "b": i*2, "c": i*3, "d": f"string_{i}"} for i in range(200000)]

async def benchmark_save_old(path):
    start = time.time()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    for row in data:
        writer.writerow(row)
    async with aiofiles.open(path, "w") as f:
        await f.write(output.getvalue())
    return time.time() - start

def _write_csv(path, data):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

async def benchmark_save_new(path):
    start = time.time()
    await asyncio.to_thread(_write_csv, path, data)
    return time.time() - start

async def benchmark_load_old(path):
    start = time.time()
    async with aiofiles.open(path, "r") as f:
        content = await f.read()
        reader = csv.DictReader(content.splitlines())
        res = [row for row in reader]
    return time.time() - start

def _read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

async def benchmark_load_new(path):
    start = time.time()
    res = await asyncio.to_thread(_read_csv, path)
    return time.time() - start

async def main():
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "1.csv")
        p2 = os.path.join(d, "2.csv")

        print(f"Save Old: {await benchmark_save_old(p1):.3f}s")
        print(f"Save New: {await benchmark_save_new(p2):.3f}s")

        print(f"Load Old: {await benchmark_load_old(p1):.3f}s")
        print(f"Load New: {await benchmark_load_new(p2):.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
