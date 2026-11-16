"""Asynchronous background worker queue for non-blocking video processing."""
from __future__ import annotations

import queue
import threading

from app.core.logging_config import logger
from app.db.repositories import update_video_progress
from app.ml.pipeline import process_video_pipeline


class VideoJobManager:
    """Single-worker background thread pool (upgradeable to Celery/Redis for multi-node scale)."""

    def __init__(self) -> None:
        self._queue: queue.Queue[int] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="video-worker")
        self._worker_thread.start()
        logger.info("Started background VideoJobManager worker.")

    def enqueue_video(self, video_id: int) -> None:
        self.start()
        self._queue.put(video_id)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                video_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                process_video_pipeline(video_id)
            except Exception as exc:
                logger.exception("Video processing failed for video_id=%s: %s", video_id, exc)
                update_video_progress(
                    video_id=video_id,
                    status_val="failed",
                    progress_pct=0.0,
                    error_message=str(exc),
                )
            finally:
                self._queue.task_done()


job_manager = VideoJobManager()
