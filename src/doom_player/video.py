"""Attempt videos: one frame per tic, streamed to an MP4 file.

Used by the MCP server for LLM Attempts and by the Eval Suite for every other
Map Contender, through `AttemptSession.act(..., on_frame=VideoWriter.add)`.
"""

from pathlib import Path

VIDEO_FPS = 35  # one frame per tic: real game speed


class VideoWriter:
    """Streams frames to an MP4 file, so a long Attempt never sits in memory."""

    def __init__(self, path: Path):
        import imageio_ffmpeg

        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio_ffmpeg.write_frames(
            str(path), (320, 240), fps=VIDEO_FPS, codec="libx264", pix_fmt_out="yuv420p", quality=7
        )
        self._writer.send(None)

    def add(self, frame) -> None:
        self._writer.send(frame.tobytes())

    def close(self) -> None:
        self._writer.close()
