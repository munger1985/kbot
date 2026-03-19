import asyncio
from datetime import datetime
from loguru import logger
from typing import Set

# Assume these are imported from your core modules
from .file_processor import FileProcessor

class FileParseEngine:
    """
    Asynchronous file parsing engine with priority queue and worker pool.
    Features:
    - Priority-based task scheduling
    - Parallel worker processing
    - Memory-based duplicate prevention
    - Graceful shutdown mechanism
    """
    def __init__(self, parallel_workers: int = 5, check_interval: int = 10):
        self.parallel_workers = parallel_workers
        self.check_interval = check_interval
        
        # 1. Priority queue: Stores tuples of (priority, timestamp, file_params)
        # Lower numeric value means higher priority
        self.queue = asyncio.PriorityQueue(maxsize=parallel_workers * 3)
        
        # 2. In-memory duplicate lock: Tracks IDs of files being processed
        self.processing_ids: Set[str] = set()
        
        # 3. Business component
        self.processor = FileProcessor()
        
        # 4. Task handles for graceful shutdown
        self.producer_task: asyncio.Task | None = None
        self.worker_tasks: list[asyncio.Task] = []

    async def _producer_loop(self):
        """Producer: Polls database for pending files and pushes to queue"""
        logger.info("Parsing producer loop will start working in 10 seconds...")
        try:
            await asyncio.sleep(10)  # Initial delay for system startup
        except asyncio.CancelledError:
            return  # Exit immediately if cancelled during initial wait
        
        logger.info("Starting parsing task producer loop...")
        while True:
            try:
                # Get pending files (ensure returns list[tuple])
                pending_files = await self.processor.get_pending_files()
                
                for priority, timestamp, file_params in pending_files:
                    file_id = file_params.file_id
                    
                    if file_id not in self.processing_ids:
                        self.processing_ids.add(file_id)
                        # Enqueue (blocks asynchronously if queue is full - backpressure implementation)
                        await self.queue.put((priority, timestamp, file_params))
                        logger.debug(f"File {file_id} enqueued successfully")
                
                # Interval between database checks
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("Producer loop cancelled, exiting...")
                break
            except Exception as e:
                logger.error(f"Error in producer loop: {e}", exc_info=True)
                # Retry after longer delay on error
                await asyncio.sleep(10)

    async def _worker_loop(self, worker_id: int):
        """Consumer: Retrieves tasks from queue and processes files"""
        logger.info(f"Worker-{worker_id} started and waiting for tasks...")
        while True:
            queue_item = None
            try:
                # Block until queue has items
                queue_item = await self.queue.get()
                priority, timestamp, file_params = queue_item
                file_id = file_params.file_id
                
                logger.info(f"Worker-{worker_id} starting parsing: {file_id} (Priority: {priority})")
                
                # Execute core parsing logic
                await self.processor.process_file(file_params)
                
                logger.info(f"Worker-{worker_id} completed parsing: {file_id}")
                
            except asyncio.CancelledError:
                logger.info(f"Worker-{worker_id} cancelled, exiting...")
                break
            except Exception as e:
                logger.error(f"Worker-{worker_id} error processing task: {e}", exc_info=True)
            finally:
                if queue_item:
                    # Release memory lock and mark task as done regardless of success/failure
                    _, _, file_params = queue_item
                    self.processing_ids.discard(file_params.file_id)
                    self.queue.task_done()

    async def start(self):
        """Start engine (called by FastAPI Lifespan)"""
        logger.info(f"Starting FileParseEngine with {self.parallel_workers} workers...")
        
        # Start consumer workers
        self.worker_tasks = [
            asyncio.create_task(self._worker_loop(i), name=f"Worker-{i}")
            for i in range(self.parallel_workers)
        ]
        
        # Start producer
        self.producer_task = asyncio.create_task(self._producer_loop(), name="Producer")
        
        logger.success("FileParseEngine started successfully")

    async def stop(self):
        """Graceful shutdown of the engine"""
        logger.warning("Initiating graceful shutdown of FileParseEngine...")
        
        # 1. Collect all tasks to cancel
        tasks_to_cancel = []
        
        # Cancel producer task
        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
            tasks_to_cancel.append(self.producer_task)
        
        # Cancel worker tasks
        for worker_task in self.worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
                tasks_to_cancel.append(worker_task)
        
        # 2. Wait for all tasks to complete cancellation
        if tasks_to_cancel:
            # Return exceptions=True to avoid raising CancelledError
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # 3. Clear processing state
        self.processing_ids.clear()
        
        # 4. Wait for remaining queue tasks to finish (optional)
        if not self.queue.empty():
            logger.info(f"Waiting for {self.queue.qsize()} remaining tasks to complete...")
            await self.queue.join()
        
        logger.success("FileParseEngine shutdown completed successfully")