"""Attempt videos, streamed to an MP4 file.

LLM Attempts (MCP server) and Door A Contenders are filmed one frame per tic,
through `AttemptSession.act(..., on_frame=VideoWriter.add)`. A learned Map
Contender is filmed one frame per decision instead (`MapEnv.on_frame`): stepping
tic by tic redraws the status bar face differently, and a policy that reads the
screen would then play a different game from the one it is scored on.
"""

from pathlib import Path

VIDEO_FPS = 35  # one frame per tic: real game speed


class VideoWriter:
    """Streams frames to an MP4 file, so a long Attempt never sits in memory."""

    def __init__(self, path: Path, fps: float = VIDEO_FPS):
        import imageio_ffmpeg

        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio_ffmpeg.write_frames(
            str(path), (320, 240), fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=7
        )
        self._writer.send(None)

    def add(self, frame) -> None:
        self._writer.send(frame.tobytes())

    def close(self) -> None:
        self._writer.close()
