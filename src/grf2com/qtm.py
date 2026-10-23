"""Start / stop / save a QTM capture over the QTM real-time protocol (qtm-rt), for recording validation mocap next to
a live log. Same sequence as the lab's Treadmill_QTM QTMControl.py: connect, take control (password as entered in
QTM), a new measurement if needed, start, and later stop and save under a take name (QTM saves it in its project's
data folder).

The asyncio loop runs in its own thread; start() / stop() / event() return concurrent.futures.Future, so a GUI never
blocks on QTM.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future


class QtmRecorder:
    def __init__(self, host: str = "127.0.0.1", password: str = "", timeout: float = 10.0):
        self.host, self.password, self.timeout = host, password, timeout
        self.conn = None
        self.capture_started_t: float | None = None
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="qtm", daemon=True).start()

    def start(self) -> Future:
        """Resolves to time.perf_counter() at QTM's capture-started event (the same clock as the live logs)."""
        return asyncio.run_coroutine_threadsafe(self._start(), self.loop)

    def event(self, label: str) -> Future:
        return asyncio.run_coroutine_threadsafe(self.conn.set_qtm_event(label), self.loop)

    def stop(self, take_name: str | None) -> Future:
        """Stops the capture and saves it as take_name (None: don't save). Resolves to the take name or None."""
        return asyncio.run_coroutine_threadsafe(self._stop(take_name), self.loop)

    @property
    def capturing(self) -> bool:
        return self.conn is not None

    async def _start(self) -> float:
        import qtm_rt

        conn = await qtm_rt.connect(self.host, timeout=5)
        if conn is None:
            raise ConnectionError(f"could not connect to QTM at {self.host}; is QTM open with its real-time "
                                  "server enabled?")
        try:
            await conn.take_control(self.password)
            if await conn.get_state() != qtm_rt.QRTEvent.EventConnected:
                await conn.new()
                await conn.await_event(qtm_rt.QRTEvent.EventConnected, timeout=self.timeout)
            await conn.start()
            await conn.await_event(qtm_rt.QRTEvent.EventCaptureStarted, timeout=self.timeout)
        except BaseException:
            await self._release(conn)
            raise
        self.conn = conn
        self.capture_started_t = time.perf_counter()
        return self.capture_started_t

    async def _stop(self, take_name: str | None) -> str | None:
        import qtm_rt

        conn, self.conn = self.conn, None
        if conn is None:
            return None
        try:
            await conn.stop()
            await conn.await_event(qtm_rt.QRTEvent.EventCaptureStopped, timeout=self.timeout)
            if take_name:
                await conn.save(take_name)
            return take_name
        finally:
            await self._release(conn)

    @staticmethod
    async def _release(conn) -> None:
        try:
            await conn.release_control()
        except Exception:  # noqa: BLE001  (already disconnected or not in control)
            pass
        conn.disconnect()
