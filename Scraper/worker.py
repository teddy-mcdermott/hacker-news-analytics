import time
import json
import os
import sys
import asyncio
import aiohttp
import logging
import asyncpg
from dotenv import load_dotenv
from pathlib import Path

# Loads environment variables from .env, make sure to set yours
from dotenv import load_dotenv
load_dotenv()  # searches upward automatically


user = os.getenv('POSTGRES_USER', 'default_user')
password = os.getenv('POSTGRES_PASSWORD', 'default_pass')
host = os.getenv('POSTGRES_HOST', 'localhost')
port = int(os.getenv('POSTGRES_PORT', '5432'))
db = os.getenv('POSTGRES_DB', 'hacker_news')

required_vars = ['POSTGRES_USER', 'POSTGRES_PASSWORD',
                 'POSTGRES_DB', 'POSTGRES_HOST', 'POSTGRES_PORT']

for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")


DB_URI = f"postgresql://{user}:{password}@{host}:{port}/{db}"

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='worker.log', level=logging.INFO)

# The batch size for DB writes and fetchs at once.
BATCH_SIZE = 1000


def log(worker_id, message):
    """Custom log function to ensure immediate,
    unbuffered output from workers."""

    if "--log" in sys.argv:
        log_message = f"[Worker {worker_id}, PID: {os.getpid()}] {message}\n"
        sys.stdout.write(log_message)
        logger.info(log_message)
        sys.stdout.flush()
    return None

# Data Storage


async def store_batch(conn, items_batch, worker_id):
    """Stores a batch of items using asyncpg for high performance."""
    if not items_batch:
        return
    columns = ['id', 'type', 'by', 'time', 'text', 'url', 'title', 'score',
               'descendants', 'parent', 'kids', 'deleted', 'dead']

    # Prepare data, ensuring kids is a JSON string
    values_to_insert = [
        tuple(json.dumps(item.get(col)) if col ==
              'kids' and item.get(col) else item.get(col) for col in columns)
        for item in items_batch if item and 'id' in item
    ]

    if not values_to_insert:
        return

    try:
        await conn.executemany(
            f"""INSERT INTO items ({', '.join(columns)})
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (id) DO NOTHING""",
            values_to_insert
        )
    except Exception as e:
        log(worker_id, f"DB Error during batch insert: {e}")


# Job Management
async def claim_job(conn, worker_id):
    """Atomically finds and claims a 'pending' job from the database."""

    # Use a transaction to ensure atomicity
    async with conn.transaction():
        job = await conn.fetchrow("""
            UPDATE job_chunks
            SET status = 'in_progress', worker_id = $1, updated_at = NOW()
            WHERE id = (
                SELECT id FROM job_chunks WHERE status = 'pending'
                ORDER BY start_id
                FOR UPDATE SKIP LOCKED LIMIT 1
            ) RETURNING id, start_id, end_id;
        """, worker_id)
    if job:
        return {'id': job['id'],
                'start_id': job['start_id'],
                'end_id': job['end_id']}
    return None


async def complete_job(conn, job_id):
    """Marks a job as 'completed' in the database."""
    await conn.execute("""UPDATE job_chunks SET status = 'completed',
                        updated_at = NOW() WHERE id = $1;", job_id""")

# API Fetching


async def fetch_item(session, item_id, worker_id):
    """Asynchronously fetches a single item,
      returning the JSON data or None."""

    url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log(worker_id, f"... {type(e).__name__}: {e}")
        return None


# Worker Logic


async def worker(worker_id):
    """The core async worker function."""
    log(worker_id, "Starting up.")
    conn = await asyncpg.connect(DB_URI)

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                job_start_time = time.monotonic()

                job = await claim_job(conn, worker_id)
                if job is None:
                    log(worker_id, "No more jobs to claim. Exiting.")
                    break

                log(worker_id, f"""Claimed job {job['id']}.
                    Range: {job['start_id']} to {job['end_id']}.""")

                # Create a list of all fetch tasks (unresolved async functions)
                tasks = [
                    fetch_item(session, item_id, worker_id)
                    for item_id in range(job['start_id'], job['end_id'] + 1)]

                results = []

                for i in range(0, len(tasks), BATCH_SIZE):
                    task_chunk = tasks[i:i+BATCH_SIZE]

                    log(worker_id,
                        f"...Job {job['id']} progress: "
                        f" fetching items {i} to {i+len(task_chunk)-1}")

                    chunk_results = await asyncio.gather(*task_chunk)

                    # Filter out None results from failed fetches
                    valid_results = [res for res in chunk_results if res]

                    if valid_results:
                        await store_batch(conn, valid_results, worker_id)
                        results.extend(valid_results)

                await complete_job(conn, job['id'])

                # Performance Calculations
                job_end_time = time.monotonic()
                duration_seconds = job_end_time - job_start_time
                items_processed = len(results)
                records_per_min = 0

                if duration_seconds > 0:
                    records_per_min = (items_processed / duration_seconds) * 60

                log(worker_id,
                    f"""Completed job {job['id']}.
                    Processed {items_processed} items.
                    Rate: {records_per_min:.2f} items/min."""
                    )

    except Exception as e:
        log(worker_id, f"An unhandled error occurred: {e}")
    finally:
        await conn.close()
        log(worker_id, "Shutting down.")


def run_worker(worker_id):
    """
    A simple synchronous wrapper to launch the async worker.
    This is what the multiprocessing Pool will call.
    """
    try:
        asyncio.run(worker(worker_id))
    except KeyboardInterrupt:
        pass
