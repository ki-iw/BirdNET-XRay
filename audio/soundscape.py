import queue
import threading
import numpy as np
import sounddevice as sd
from enum import Enum
from pydub import AudioSegment

# ====== Config ======
SAMPLE_RATE = 44100
CHANNELS = 2
BLOCKSIZE = 1024  # Lower = lower latency, higher CPU
SAMPLE_RATE = 48000      
CHANNELS    = 1          #mono
BLOCKSIZE   = 1024

def seg_to_float32(seg: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to float32 numpy array (shape: [frames, channels])."""
    # Standardize
    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
    # Convert to numpy float32 in [-1, 1]
    samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
    #if CHANNELS == 2:
    #    samples = samples.reshape((-1, 2))
    #else:
    #    samples = samples.reshape((-1, 1))
    samples = samples.reshape((-1, CHANNELS if CHANNELS > 1 else 1))
    return (samples.astype(np.float32) / 32768.0)
    # 16-bit -> float
    

def db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)

class CLIP_TYPE(Enum):
    AMBIANCE = 1
    SOUND = 2
    LIVE = 3

class Clip:
    def __init__(self, path: str, clip_id: int, clip_type: CLIP_TYPE, data: np.ndarray, start_sample: int, gain: float = 1.0, fade_in: int = 0, fade_out: int = 0):
        """
        data: float32 array [frames, channels]
        start_sample: absolute sample index in the output timeline when this clip should start
        gain: linear gain (1.0 = 0 dB)
        fade_in/fade_out: samples for fade (optional)
        """
        self.path = path
        self.clip_id = clip_id
        self.type = clip_type
        self.data = data
        self.start = start_sample
        self.gain = gain
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.length = data.shape[0]

    def render_chunk(self, global_start: int, nframes: int) -> np.ndarray:
        """
        Return this clip's contribution for the output block starting at global_start (absolute sample index),
        length nframes. Shape [nframes, channels].
        """
        end = global_start + nframes

        # Compute overlap of this output window with the clip's time window
        clip_start = self.start
        clip_end = self.start + self.length
        ov_start = max(global_start, clip_start)
        ov_end = min(end, clip_end)
        if ov_end <= ov_start:
            return None  # no overlap

        # Indices into the output block
        out_s = ov_start - global_start
        out_e = ov_end - global_start

        # Indices into the clip buffer
        clip_s = ov_start - clip_start
        clip_e = ov_end - clip_start

        buf = np.zeros((nframes, self.data.shape[1]), dtype=np.float32)
        chunk = self.data[clip_s:clip_e].copy()

        # Apply fades if configured
        if self.fade_in > 0:
            fi_s = clip_s
            fi_e = min(clip_e, self.fade_in)
            if fi_e > fi_s:
                ramp = np.linspace(fi_s / self.fade_in, (fi_e - 1) / self.fade_in, fi_e - fi_s).reshape(-1, 1)
                chunk[:fi_e - fi_s] *= ramp

        if self.fade_out > 0:
            # fade out over the last fade_out samples of the clip
            fo_region_start = self.length - self.fade_out
            if fo_region_start < self.length:
                # overlap with current chunk
                fo_s = max(clip_s, fo_region_start)
                fo_e = clip_e
                if fo_e > fo_s:
                    # position within fade region
                    pos = np.arange(fo_s, fo_e) - fo_region_start
                    ramp = (1.0 - pos / max(1, self.fade_out)).reshape(-1, 1)
                    chunk[(fo_s - clip_s):(fo_e - clip_s)] *= ramp

        buf[out_s:out_e] = chunk * self.gain
        return buf

class SoundscapeMixer:
    def __init__(self, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCKSIZE, max_samples: int = 144000):
        self.sr = samplerate
        self.ch = channels
        self.blocksize = blocksize
        self._paths = []
        self._clips = []               # active clips
        self._add_queue = queue.Queue()# thread-safe additions
        self._sample_clock = 0         # absolute samples played so far
        self._lock = threading.Lock()

        self._rb_size = max(1, max_samples)                 # total scalar samples (frames*channels)
        self._rb = np.zeros(self._rb_size, dtype=np.float32)
        self._rb_write = 0                                  # write position (in samples, interleaved)
        self._rb_count = 0                                  # how many valid samples currently buffered
        self._rb_lock = threading.Lock()

        self._next_clip_id = 1
        self._known_clip_ids = set()
        self._cancelled_ids = set()

        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.ch,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._callback
        )

    def _write_ring(self, block_2d: np.ndarray):
        """Write a block shaped (frames, channels) into the interleaved ring buffer."""
        flat = block_2d.reshape(-1)  # interleaved
        n = flat.shape[0]            # number of scalar samples being written
        with self._rb_lock:
            # If the block is larger than the ring, keep only the last part
            if n >= self._rb_size:
                flat = flat[-self._rb_size:]
                n = flat.shape[0]

            end_space = self._rb_size - self._rb_write
            if n <= end_space:
                self._rb[self._rb_write:self._rb_write + n] = flat
            else:
                # wrap
                self._rb[self._rb_write:] = flat[:end_space]
                self._rb[:n - end_space] = flat[end_space:]

            self._rb_write = (self._rb_write + n) % self._rb_size
            self._rb_count = min(self._rb_count + n, self._rb_size)

    def _read_ring(self, requested_samples: int) -> np.ndarray:
        """Read up to requested_samples (scalar samples) as a 1-D interleaved float32 array."""
        with self._rb_lock:
            n = min(requested_samples, self._rb_count)
            if n <= 0:
                return np.array([], dtype=np.float32)
            start = (self._rb_write - n) % self._rb_size
            if start + n <= self._rb_size:
                return self._rb[start:start + n].copy()
            else:
                first = self._rb[start:]
                second = self._rb[: (start + n) % self._rb_size]
                return np.concatenate((first, second), axis=0)

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def _callback(self, outdata, frames, time, status):
        if status:
            # Underruns/overruns etc.
            pass

        # Drain queued additions so they become active this block
        while True:
            try:
                clip = self._add_queue.get_nowait()
                with self._lock:
                    self._clips.append(clip)
            except queue.Empty:
                break

        # Mix
        mix = np.zeros((frames, self.ch), dtype=np.float32)
        remove_idx = []
        with self._lock:
            for i, c in enumerate(self._clips):
                contrib = c.render_chunk(self._sample_clock, frames)
                if contrib is not None:
                    mix += contrib
                # If the clip is entirely finished before the end of THIS block, mark for removal
                if self._sample_clock + frames >= c.start + c.length:
                    remove_idx.append(i)
            # Clean up finished clips
            for idx in reversed(remove_idx):
                self._clips.pop(idx)

        # Simple soft limiter to avoid hard clipping
        """ peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix /= peak """

        self._write_ring(mix)

        outdata[:] = mix
        self._sample_clock += frames

    def add_clip(self, path: str, clip_type: CLIP_TYPE = CLIP_TYPE.AMBIANCE, when_seconds: float = 0.0, gain_db: float = 0.0,
                 fade_in_ms: int = 0, fade_out_ms: int = 0) -> int:
        """Schedule an audio file to play at now + when_seconds."""

        if clip_type == CLIP_TYPE.SOUND and path in self._paths:
            print("Sound already in paths, toggling sound...")
            self.remove_clip_path(path)
            self._paths.remove(path)
            return -1

        seg = AudioSegment.from_file(path)

        if fade_in_ms > 0:
            seg = seg.fade_in(fade_in_ms)
        if fade_out_ms > 0:
            seg = seg.fade_out(fade_out_ms)

        data = seg_to_float32(seg)
        gain = db_to_gain(gain_db)
        # Convert times to samples
        with self._lock:
            clip_id = self._next_clip_id
            self._next_clip_id += 1
            start_sample = self._sample_clock + int(when_seconds * self.sr)
            self._known_clip_ids.add(clip_id)
        clip = Clip(
            path=path,
            clip_id=clip_id,
            clip_type=clip_type,
            data=data,
            start_sample=start_sample,
            gain=gain,
            fade_in=int(fade_in_ms * self.sr / 1000),
            fade_out=int(fade_out_ms * self.sr / 1000)
        )
        
        if clip_type == CLIP_TYPE.SOUND:
            self._paths.append(path)

        self._add_queue.put(clip)

        return clip_id

    def remove_clip(self, clip_id: int) -> bool:
        """
        Cancel a scheduled or currently playing clip.
        Returns True if the id was known (queued or active), False otherwise.
        Safe to call while audio is running.
        """
        with self._lock:
            known = clip_id in self._known_clip_ids
            self._cancelled_ids.add(clip_id)
            # prune immediately from active list
            if self._clips:
                self._clips = [c for c in self._clips if c.clip_id != clip_id]
            return known

    def remove_clip_path(self, path: str) -> bool:
        """
        Cancel a scheduled or currently playing clip based on path.
        Returns True if the id was known (queued or active), False otherwise.
        Safe to call while audio is running.
        """
        with self._lock:
            known = path in self._paths
            # prune immediately from active list
            if self._clips:
                self._clips = [c for c in self._clips if c.path != path]
            return known

    def clear_clips(self):
        with self._lock:
            self._clips = []

    def get_current_samples(self, interleaved: bool = False, max_samples: int = 144000) -> np.ndarray:
        """
        Return up to max_samples of the most recently played audio as float32.
        - If interleaved=False: returns shape (frames, channels).
        - If interleaved=True: returns shape (N,), interleaved.
        Note: max_samples counts *scalar* samples (frames*channels).
        """
        data_1d = self._read_ring(max_samples)
        if interleaved:
            return data_1d
        # reshape to (frames, channels)
        if data_1d.size == 0:
            return np.array([], dtype=np.float32)
        frames = data_1d.size // self.ch
        data_1d = data_1d[:frames * self.ch]          # truncate to full frames
        return data_1d.reshape(frames, self.ch).astype(np.float32, copy=False)
    
    def get_current_samples_144k(self) -> np.ndarray:
        """
        Returns exactly 144,000 scalar samples (float32, mono).
        If fewer available, left-pads with zeros.
        Shape: (144000,)
        """
        data = self._read_ring(144000)  # interleaved 1-D, but we're mono
        if data.size >= 144000:
            return data[-144000:]
        if data.size == 0:
            return np.zeros(144000, dtype=np.float32)
        pad = np.zeros(144000 - data.size, dtype=np.float32)
        return np.concatenate([pad, data], axis=0)
    
    def get_active_clips(self) -> set:
        return self._known_clip_ids

# ====== Example usage ======
if __name__ == "__main__":
    mixer = SoundscapeMixer()
    mixer.start()
    print("Soundscape running. Press Ctrl+C to stop.")

    try:
        # Start a base ambience loop immediately (e.g., a long rainforest.mp3)
        mixer.add_clip("data/clips/forest-ambience.mp3", when_seconds=0.0, gain_db=-6, fade_in_ms=3000)

        # Drop a bird call at 5s, a distant thunder at 12.3s, and footsteps now
        mixer.add_clip("data/clips/bird-call-in-spring.mp3", when_seconds=5.0, gain_db=-2)
        mixer.add_clip("data/clips/thunder-sound.mp3", when_seconds=12.3, gain_db=-4, fade_in_ms=50, fade_out_ms=500)
        mixer.add_clip("data/clips/concrete-footsteps.mp3", when_seconds=0.0, gain_db=-8)

        # You can add more sounds dynamically from other threads,
        # user input, OSC/MIDI events, etc.
        while True:
            # Example: read commands from the console to add clips live
            cmd = input("add <path> <when_s> <gain_db> [fade_in_ms] [fade_out_ms] or 'quit': ").strip()
            if cmd == "quit":
                break
            try:
                parts = cmd.split()
                path = parts[1]
                when_s = float(parts[2])
                gain_db = float(parts[3])
                fi = int(parts[4]) if len(parts) > 4 else 0
                fo = int(parts[5]) if len(parts) > 5 else 0
                mixer.add_clip(path, when_seconds=when_s, gain_db=gain_db, fade_in_ms=fi, fade_out_ms=fo)
            except Exception as e:
                print("Usage error:", e)
    except KeyboardInterrupt:
        pass
    finally:
        mixer.stop()
