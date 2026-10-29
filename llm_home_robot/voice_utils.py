"""Voice helpers: energy-based voice activity detection, transcript clean-up,
wake-word handling and text-to-speech. Pure Python + numpy (no ROS)."""
import re
import shutil
import subprocess

import numpy as np

# Whisper sometimes "hears" these in silence or noise
HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "bye", "bye bye", "okay", "ok", "so", "um", "uh", "hmm",
    "subtitles by the amara org community", "please subscribe", "music",
}


class Segmenter:
    """Cuts a stream of audio frames into utterances.

    feed(frame) -> numpy array of the finished utterance, or None.
    The first `calib_frames` frames measure the background noise; speech starts
    when the RMS exceeds max(min_rms, ratio * noise) for `start_frames` frames
    and ends after `silence_s` of quiet (or `max_s`)."""

    def __init__(self, rate=16000, frame_ms=30, min_rms=0.01, ratio=3.0, start_frames=3,
                 silence_s=0.8, max_s=10.0, preroll_s=0.3, calib_frames=20, min_speech_s=0.3):
        self.rate, self.frame = rate, int(rate * frame_ms / 1000)
        self.min_rms, self.ratio, self.start_frames = min_rms, ratio, start_frames
        self.silence_frames = int(silence_s * 1000 / frame_ms)
        self.max_frames = int(max_s * 1000 / frame_ms)
        self.preroll = int(preroll_s * 1000 / frame_ms)
        self.min_speech = int(min_speech_s * 1000 / frame_ms)
        self.calib_frames = calib_frames
        self.noise, self.n_calib = 0.0, 0
        self.reset()

    def reset(self):
        self.buf, self.recent, self.active = [], [], False
        self.voiced_run, self.quiet_run, self.voiced_total = 0, 0, 0

    @property
    def threshold(self):
        return max(self.min_rms, self.ratio * self.noise)

    def feed(self, frame):
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(frame ** 2))) if frame.size else 0.0
        if self.n_calib < self.calib_frames:
            self.n_calib += 1
            self.noise += (rms - self.noise) / self.n_calib
            return None
        voiced = rms > self.threshold
        if not self.active:
            if not voiced:     # slowly track background noise
                self.noise = 0.98 * self.noise + 0.02 * rms
            self.recent.append(frame)
            self.recent = self.recent[-(self.preroll + self.start_frames):]
            self.voiced_run = self.voiced_run + 1 if voiced else 0
            if self.voiced_run >= self.start_frames:
                self.active, self.buf = True, list(self.recent)
                self.quiet_run, self.voiced_total = 0, self.voiced_run
            return None
        self.buf.append(frame)
        if voiced:
            self.quiet_run, self.voiced_total = 0, self.voiced_total + 1
        else:
            self.quiet_run += 1
        if self.quiet_run >= self.silence_frames or len(self.buf) >= self.max_frames:
            audio, long_enough = np.concatenate(self.buf), self.voiced_total >= self.min_speech
            self.reset()
            return audio if long_enough else None
        return None


def clean_transcript(text, wake_word=""):
    """Normalise Whisper output. Returns the command text, or None to ignore.
    With a wake word ("robot"), only utterances containing it are accepted and
    everything up to and including it is removed."""
    t = text.strip()
    t = re.sub(r"\[.*?\]|\(.*?\)", " ", t)            # [Music], (noise)
    t = re.sub(r"\s+", " ", t).strip()
    bare = re.sub(r"[^a-z' ]", "", t.lower()).strip()
    if bare in HALLUCINATIONS or not re.search(r"[a-z]", bare):
        return None
    if wake_word:
        ww = re.escape(wake_word.lower())
        m = re.search(rf"\b(?:hey |ok |okay )?{ww}\b[,.!?]?\s*", t.lower())
        if not m:
            return None
        t = t[m.end():].strip()
        if not t:
            return None
    return t.strip(" .")


def whisper_prompt(places):
    """A hint so Whisper spells our vocabulary right."""
    names = ", ".join(p.replace("_", " ") for p in places)
    return (f"Commands for a home robot. Places: {names}. "
            "Go to the kitchen. Follow me. Find the red ball. Stop. Turn left 90 degrees. "
            "Move forward 2 meters. What do you see? Patrol the house.")


class Speaker:
    """Text-to-speech through whatever is installed: spd-say, espeak-ng, espeak."""

    def __init__(self, enabled=True):
        self.cmd = None
        if enabled:
            for exe, args in (("spd-say", ["-w", "-r", "-5"]), ("espeak-ng", ["-s", "165"]),
                              ("espeak", ["-s", "165"])):
                if shutil.which(exe):
                    self.cmd = [exe, *args]
                    break

    @property
    def available(self):
        return self.cmd is not None

    def say(self, text):
        if not self.cmd:
            return
        text = text.replace("°", " degrees").replace(" m ", " metres ").replace(" m)", " metres)")
        try:
            subprocess.run([*self.cmd, text], timeout=30, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            pass
