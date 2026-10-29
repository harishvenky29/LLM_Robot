"""Talk to the robot: microphone -> Whisper -> /robot/command, and the robot's
replies (/robot/reply) are spoken out loud.

    ros2 run llm_home_robot voice                                  # hands-free
    ros2 run llm_home_robot voice --ros-args -p wake_word:=robot   # only "robot, go to the kitchen"
    ros2 run llm_home_robot voice --ros-args -p mode:=ptt          # press Enter to talk
    ros2 run llm_home_robot voice --ros-args -p model:=small.en    # more accurate, slower
    ros2 run llm_home_robot voice --ros-args -p wav:=command.wav   # test without a mic

Needs:  pip install faster-whisper sounddevice --break-system-packages
        sudo apt install libportaudio2 speech-dispatcher   (or espeak-ng)
Whisper runs locally (base.en ~150 MB, downloaded once). The mic is muted
while the robot speaks so it doesn't hear itself.
"""
import os
import queue
import sys
import threading
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from .place_graph import PlaceGraph
from .voice_utils import Segmenter, Speaker, clean_transcript, whisper_prompt

RATE = 16000
FRAME_MS = 30


class Voice(Node):
    def __init__(self):
        super().__init__("voice")
        share = get_package_share_directory("llm_home_robot")
        p = self.declare_parameter
        p("model", "base.en")          # tiny.en | base.en | small.en | medium.en
        p("mode", "vad")               # vad (hands-free) | ptt (press Enter)
        p("wake_word", "")             # e.g. "robot"; empty = every utterance is a command
        p("speak", True)               # read replies aloud
        p("device", -1)                # microphone index (-1 = default); list: python3 -m sounddevice
        p("whisper_device", "cpu")     # cpu (works everywhere) | cuda (needs NVIDIA CUDA 12 + cuDNN)
        p("compute_type", "int8")      # int8 on CPU, float16 on a GPU
        p("min_rms", 0.01)             # raise in a noisy room
        p("wav", "")                   # transcribe this file once instead of using the mic
        p("places_file", os.path.join(share, "config", "places.yaml"))
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode, self.wake = gp("mode"), gp("wake_word").strip()
        self.device = None if gp("device") < 0 else gp("device")
        self.min_rms, self.wav = gp("min_rms"), gp("wav")
        self.prompt = whisper_prompt(PlaceGraph.from_files(gp("places_file")).place_names())

        try:
            if not self.wav:
                import sounddevice  # noqa: F401
        except (ImportError, OSError) as e:
            raise SystemExit(f"Microphone library missing ({e}).\n"
                             "  pip install sounddevice --break-system-packages\n"
                             "  sudo apt install libportaudio2") from e
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise SystemExit("Whisper missing: pip install faster-whisper --break-system-packages") from e
        self.get_logger().info(f"Loading Whisper '{gp('model')}' (first run downloads it)...")
        self.WhisperModel, self.model_name = WhisperModel, gp("model")
        self.model = WhisperModel(gp("model"), device=gp("whisper_device"), compute_type=gp("compute_type"))

        self.speaker = Speaker(gp("speak"))
        if gp("speak") and not self.speaker.available:
            self.get_logger().warn("No text-to-speech found: sudo apt install speech-dispatcher (or espeak-ng)")
        self.speaking = threading.Event()
        self.replies = queue.Queue()
        self.pub = self.create_publisher(String, "/robot/command", 10)
        self.create_subscription(String, "/robot/reply", lambda m: self.replies.put(m.data), 10)
        threading.Thread(target=self.speak_loop, daemon=True).start()

    # ---------------------------------------------------------------- output
    def speak_loop(self):
        while True:
            text = self.replies.get()
            print(f"\r\033[K🤖 {text}")
            if text.startswith("Thinking"):
                continue
            self.speaking.set()
            self.speaker.say(text)
            time.sleep(0.3)
            self.speaking.clear()

    # ----------------------------------------------------------------- input
    def transcribe(self, audio):
        try:
            segs = self._transcribe(audio)
        except RuntimeError as e:
            # GPU chosen but the NVIDIA libraries (libcublas / cuDNN) are missing: use the CPU
            if "cuda" not in str(e).lower() and "cublas" not in str(e).lower() and "cudnn" not in str(e).lower():
                raise
            self.get_logger().warn(f"Whisper can't use the GPU ({e}) - switching to the CPU.")
            self.model = self.WhisperModel(self.model_name, device="cpu", compute_type="int8")
            segs = self._transcribe(audio)
        if not segs or min(s.no_speech_prob for s in segs) > 0.6:
            return ""
        return " ".join(s.text for s in segs)

    def _transcribe(self, audio):
        segments, _info = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False, initial_prompt=self.prompt)
        return list(segments)

    def on_utterance(self, audio):
        t0 = time.time()
        raw = self.transcribe(audio)
        text = clean_transcript(raw, self.wake)
        if text is None:
            if raw.strip():
                self.get_logger().debug(f"ignored: {raw!r}")
            return
        print(f"\r\033[K🎤 \"{text}\"  ({time.time() - t0:.1f}s)")
        self.pub.publish(String(data=text))

    def run_vad(self):
        import sounddevice as sd
        seg = Segmenter(RATE, FRAME_MS, min_rms=self.min_rms)
        frames = queue.Queue()
        wake = f' for "{self.wake}"' if self.wake else ""
        print(f"\n🎤 Listening{wake}... "
              "(stay quiet for a second while I measure the room noise; Ctrl+C to quit)")
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=self.device,
                            blocksize=int(RATE * FRAME_MS / 1000),
                            callback=lambda d, *_: frames.put(d[:, 0].copy())):
            while rclpy.ok():
                frame = frames.get()
                if self.speaking.is_set():       # don't listen to ourselves
                    seg.reset()
                    continue
                audio = seg.feed(frame)
                if audio is not None:
                    self.on_utterance(audio)

    def run_wav(self):
        import wave
        with wave.open(self.wav) as w:
            sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
            a = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
        a = a.reshape(-1, ch).mean(axis=1)
        if sr != RATE:
            a = np.interp(np.arange(0, len(a), sr / RATE), np.arange(len(a)), a).astype(np.float32)
        print(f"🎤 (from {self.wav})")
        self.on_utterance(a)
        time.sleep(2.0)                       # let the reply arrive and be spoken
        while self.speaking.is_set() or not self.replies.empty():
            time.sleep(0.2)

    def run_ptt(self):
        import sounddevice as sd
        print("\n🎤 Press Enter, speak, press Enter again. (Ctrl+C to quit)")
        while rclpy.ok():
            input("   [Enter] to talk ")
            chunks = []
            with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=self.device,
                                callback=lambda d, *_: chunks.append(d[:, 0].copy())):
                input("   ...listening, [Enter] to stop ")
            if chunks:
                self.on_utterance(np.concatenate(chunks))


def main():
    rclpy.init()
    try:
        node = Voice()
    except SystemExit as e:
        print(e, file=sys.stderr)
        rclpy.shutdown()
        return
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        if node.wav:
            node.run_wav()
        elif node.mode == "ptt":
            node.run_ptt()
        else:
            node.run_vad()
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as e:  # noqa: BLE001
        if "sounddevice" not in type(e).__module__ and "PortAudio" not in type(e).__name__:
            print(f"\nVoice error: {type(e).__name__}: {e}", file=sys.stderr)
            raise
        print(f"\nMicrophone problem: {e}\n"
              "  - check a mic is plugged in and not muted (Settings > Sound > Input)\n"
              "  - list devices:  python3 -m sounddevice\n"
              "  - pick one:      ros2 run llm_home_robot voice --ros-args -p device:=<index>",
              file=sys.stderr)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.stdout.flush()
        os._exit(0)                  # skip Whisper's C++ thread teardown (it can abort on exit)


if __name__ == "__main__":
    main()
