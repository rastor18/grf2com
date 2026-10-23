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
import xml.etree.ElementTree as ET
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


def read_force_plates(host: str = "127.0.0.1", timeout: float = 3.0) -> list[dict]:
    """The force plates in QTM's settings, read-only (no control or password needed): [{name, corners, origin}],
    corners (4, 3) in m in the lab frame, origin (3,) in m in the plate's own axes (QTM's Origin setting)."""
    import qtm_rt

    async def get() -> bytes:
        conn = await qtm_rt.connect(host, timeout=timeout)
        if conn is None:
            raise ConnectionError(f"could not connect to QTM at {host}")
        try:
            return await conn.get_parameters(parameters=["force"])
        finally:
            conn.disconnect()

    force = ET.fromstring(asyncio.run(get())).find("Force")
    if force is None:
        raise LookupError("QTM has no measurement or preview open, so its force-plate settings aren't available")
    scale = 0.001 if (force.findtext("Unit_Length") or "mm").strip() == "mm" else 1.0
    plates = []
    for plate in force.findall("Plate"):
        loc = plate.find("Location")
        corners = [[scale * float(loc.find(f"Corner{i}").findtext(a)) for a in "XYZ"] for i in range(1, 5)]
        origin = plate.find("Origin")
        plates.append({"name": plate.findtext("Name") or f"plate {plate.findtext('Force_Plate_Index')}",
                       "corners": corners,
                       "origin": [scale * float(origin.findtext(a)) for a in "XYZ"] if origin is not None else None})
    if not plates:
        raise ValueError("QTM has no force plates configured")
    return plates
