import tkinter as tk
import cv2
import numpy as np
import datetime
import multiprocessing
import time
import csv
import os
import json
from tkinter import simpledialog
import tkinter
from pathlib import Path
import logging

# prevent ghost root window
tkinter.Tk().withdraw()

# ===================== CONFIG =====================
ALL_PATTERNS = ["grid", "circle", "line", "random"]
GLOBAL_METADATA_PATH = os.path.join("sessions", "metadata.json")
CAPTURE_FPS = 30                 # VideoWriter FPS
PNG_COMPRESS = 3                 # 0..9 (lower=faster)
STATE_DELAY_FRAMES = 1           # apply fixation change starting next frame
DEBUG = True                     # set to False to silence logs
# ===================================================

# --- add near the top with your other imports ---
import threading, queue

# ====== FAST SAVE SETTINGS ======
SAVE_FORMAT = "jpg"          # "jpg" (fast) or "png" (slower)
JPEG_QUALITY = 90            # 1..100 (higher = better, larger)
PNG_COMPRESS = 0             # 0..9  (lower = faster)
SAVE_EVERY_N = 1             # 1 = save every frame, 2 = every other, etc.
SAVE_QUEUE_MAX = 2000        # how many frames can wait to be saved
# ================================

class FrameSaver:
    """Background disk writer to avoid blocking the camera loop."""
    def __init__(self, maxsize=SAVE_QUEUE_MAX):
        self.q = queue.Queue(maxsize=maxsize)
        self.stop_flag = threading.Event()
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _worker(self):
        while not self.stop_flag.is_set() or not self.q.empty():
            try:
                out_path, frame = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                ext = os.path.splitext(out_path)[1].lower()
                if ext == ".png":
                    ok = cv2.imwrite(out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, int(PNG_COMPRESS)])
                else:
                    ok = cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])
                if not ok:
                    print(f"❌ cv2.imwrite failed: {out_path}")
            finally:
                self.q.task_done()

    def save(self, out_path, frame):
        # block a tiny bit rather than dropping frames
        try:
            self.q.put((out_path, frame.copy()), timeout=0.25)
        except queue.Full:
            # If disk can’t keep up, you can choose to drop or block longer.
            print("⚠️ Save queue full — dropping a frame to keep capture real-time.")

    def close(self):
        self.q.join()
        self.stop_flag.set()
        self.t.join(timeout=2)


def dbg(*a):
    if DEBUG:
        print(*a)

def ensure_metadata_file():
    os.makedirs("sessions", exist_ok=True)
    if not os.path.exists(GLOBAL_METADATA_PATH):
        with open(GLOBAL_METADATA_PATH, "w") as f:
            json.dump({}, f, indent=2)

# --- Logging (helps verify folder creation) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")
log = logging.getLogger("frames")

# --- Helpers to guarantee folder structure ---
def precreate_pattern_dirs(trial_folder: str, patterns):
    """
    Create <trial_folder>/<pattern>/frames for each pattern *before* anything starts.
    Also drops a .keep so the folder is never optimized away by tools.
    """
    for pattern in patterns:
        pattern_folder = os.path.join(trial_folder, pattern.replace(" ", "_"))
        frames_dir = os.path.join(pattern_folder, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        keep = os.path.join(frames_dir, ".keep")
        try:
            Path(keep).touch(exist_ok=True)
        except Exception as e:
            dbg(f"⚠️ couldn't create .keep in {frames_dir}: {e}")
        log.info("Frames dir ready → %s", Path(frames_dir).resolve())

# ======== REPLACE YOUR camera_loop WITH THIS ONE ========
def camera_loop(recording_flag, stop_event, video_path, video_start_ns,
                trial_folder, shared_state):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera"); return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = CAPTURE_FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("❌ Cannot open VideoWriter"); cap.release(); return

    print(f"🎥 Recording session to {video_path}")
    first_written = False
    global_frame_idx = 0

    frames_dir_cache = {}

    # state machine
    effective_pattern = None
    effective_fix_idx = -1
    pending_state = None
    apply_at_idx = -1

    # manifest for quick QA
    manifest = {}

    # new: async saver
    saver = FrameSaver()

    def get_frames_dir(pattern_name):
        if pattern_name in frames_dir_cache:
            return frames_dir_cache[pattern_name]
        pattern_folder = os.path.join(trial_folder, pattern_name.replace(" ", "_"))
        frames_dir = os.path.join(pattern_folder, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frames_dir_cache[pattern_name] = frames_dir
        return frames_dir

    def bump_manifest(pat, fix, idx):
        dpat = manifest.setdefault(pat, {})
        dfix = dpat.setdefault(int(fix), {"count": 0, "first": None, "last": None})
        dfix["count"] += 1
        if dfix["first"] is None or idx < dfix["first"]:
            dfix["first"] = idx
        if dfix["last"] is None or idx > dfix["last"]:
            dfix["last"] = idx

    # choose extension based on SAVE_FORMAT
    def ext_for_format():
        return ".png" if SAVE_FORMAT.lower() == "png" else ".jpg"

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                dbg("⚠️ cap.read() failed; stopping.")
                break

            # preview (this can also slow the loop; close the window to test max throughput)
            cv2.imshow("Camera Preview (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            if recording_flag.value:
                out.write(frame)
                if not first_written:
                    video_start_ns.value = time.time_ns()
                    dbg(f"⏱️ video_start_ns set to {video_start_ns.value}")
                    first_written = True

                curr_idx = global_frame_idx

                # 1) apply if due
                if pending_state is not None and curr_idx >= apply_at_idx:
                    effective_pattern, effective_fix_idx = pending_state
                    dbg(f"✅ state applied at frame {curr_idx}: pattern={effective_pattern}, fix={effective_fix_idx}")
                    pending_state = None

                # 2) read shared state
                shared_pat = shared_state.get("pattern", None)
                shared_fix = int(shared_state.get("fix_idx", -1))

                # 3) schedule only if nothing pending
                if pending_state is None and ((shared_pat != effective_pattern) or (shared_fix != effective_fix_idx)):
                    pending_state = (shared_pat, shared_fix)
                    apply_at_idx = curr_idx + max(1, int(STATE_DELAY_FRAMES))
                    dbg(f"↪️  pending_state={pending_state} will apply at frame {apply_at_idx}")

                # 4) save frames during fixation (async)
                if effective_pattern and effective_fix_idx >= 0 and (curr_idx % SAVE_EVERY_N == 0):
                    frames_dir = get_frames_dir(effective_pattern)
                    out_path = os.path.join(
                        frames_dir,
                        f"{int(effective_fix_idx):02d}_{curr_idx:06d}{ext_for_format()}"
                    )
                    saver.save(out_path, frame)
                    if global_frame_idx % 30 == 0:
                        dbg(f"💾 queued save {os.path.basename(out_path)}")
                    bump_manifest(effective_pattern, effective_fix_idx, curr_idx)

                global_frame_idx += 1
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # make sure all queued frames are written
        saver.close()
        print("📹 Camera recording stopped.")

        # write manifests
        for pat, stats in manifest.items():
            pattern_folder = os.path.join(trial_folder, pat.replace(" ", "_"))
            os.makedirs(pattern_folder, exist_ok=True)
            manifest_path = os.path.join(pattern_folder, "frames_manifest.csv")
            with open(manifest_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["fixation_index", "count", "first_frame_idx", "last_frame_idx"])
                for fix_idx in sorted(stats.keys()):
                    entry = stats[fix_idx]
                    w.writerow([fix_idx, entry["count"], entry["first"], entry["last"]])
            print(f"📝 Wrote manifest: {manifest_path}")
# ======== end camera_loop ========


def launch_gaze_target(shared_event_logs, trial_folder, stop_event, shared_r_triggers, shared_state):
    def run_gui():
        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.configure(bg="#C8C8C8")
        canvas = tk.Canvas(root, bg="#C8C8C8", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        target_radius = 10
        fps = 60
        frame_delay = 1 / fps
        pattern_index = 0

        def draw_target(x, y, radius):
            canvas.delete("all")
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#646464", outline="#646464")
            cross_len = radius * 0.6
            thickness = 2
            canvas.create_line(x - cross_len, y, x + cross_len, y, fill="white", width=thickness)
            canvas.create_line(x, y - cross_len, x, y + cross_len, fill="white", width=thickness)
            canvas.update()

        def animate_to(x0, y0, x1, y1, duration=0.3):
            steps = max(1, int(duration * fps))
            for i in range(steps):
                if stop_event.is_set():
                    return
                t = i / max(1, steps - 1)
                x, y = int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)
                draw_target(x, y, target_radius)
                time.sleep(frame_delay)

        def get_fixation_positions(pattern):
            if pattern == "grid":
                return [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
                        (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
                        (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]
            elif pattern == "circle":
                cx, cy, r = 0.5, 0.5, 0.3
                return [(cx + r * np.cos(2 * np.pi * i / 10), cy + r * np.sin(2 * np.pi * i / 10)) for i in range(10)]
            elif pattern == "line":
                return [(0.1 + 0.08 * i, 0.5) for i in range(10)]
            elif pattern == "random":
                np.random.seed(42)
                return list(zip(np.random.rand(5), np.random.rand(5)))
            return []

        def run_pattern(pattern_name):
            rel_positions = get_fixation_positions(pattern_name)
            pattern_folder = os.path.join(trial_folder, pattern_name.replace(" ", "_"))
            frames_dir = os.path.join(pattern_folder, "frames")

            # Hard guarantees for folders every time a pattern starts
            os.makedirs(pattern_folder, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)
            log.info("Pattern start; frames dir confirmed → %s", Path(frames_dir).resolve())

            log_path = os.path.join(pattern_folder, "event_log.csv")
            coord_path = os.path.join(pattern_folder, "fixation_coords.csv")

            with open(log_path, "w", newline="") as log_file, open(coord_path, "w", newline="") as coord_file:
                log_writer = csv.writer(log_file)
                coord_writer = csv.writer(coord_file)
                log_writer.writerow(["event", "timestamp_ns"])
                coord_writer.writerow(["fixation_index", "x_px", "y_px", "rel_x", "rel_y"])

                ts_pattern_start = time.time_ns()
                log_writer.writerow(["pattern_start", ts_pattern_start])

                # announce current pattern; no fixation yet
                shared_state["pattern"] = pattern_name
                shared_state["fix_idx"] = -1
                dbg(f"🎯 Pattern '{pattern_name}' started")

                current_x, current_y = int(screen_width * 0.5), int(screen_height * 0.5)
                draw_target(current_x, current_y, target_radius)

                for idx, (rel_x, rel_y) in enumerate(rel_positions):
                    next_x, next_y = int(screen_width * rel_x), int(screen_height * rel_y)
                    animate_to(current_x, current_y, next_x, next_y)
                    draw_target(next_x, next_y, target_radius)

                    ts_fix_start = time.time_ns()
                    log_writer.writerow(["fixation_start", ts_fix_start])
                    coord_writer.writerow([idx, next_x, next_y, round(rel_x, 3), round(rel_y, 3)])

                    shared_state["fix_idx"] = int(idx)  # enter fixation
                    dbg(f"🟢 fix {idx:02d} START")

                    time.sleep(2)  # hold fixation

                    ts_fix_end = time.time_ns()
                    log_writer.writerow(["fixation_end", ts_fix_end])

                    shared_state["fix_idx"] = -1  # leave fixation
                    dbg(f"🔴 fix {idx:02d} END")

                    current_x, current_y = next_x, next_y

                time.sleep(2)
                ts_pattern_end = time.time_ns()
                log_writer.writerow(["pattern_end", ts_pattern_end])

            shared_event_logs[pattern_name] = log_path

            # clear pattern after finishing
            shared_state["pattern"] = None
            shared_state["fix_idx"] = -1
            dbg(f"🏁 Pattern '{pattern_name}' finished")

        def show_instruction():
            if pattern_index >= len(ALL_PATTERNS):
                canvas.delete("all")
                root.quit()
                return
            canvas.delete("all")
            canvas.create_text(screen_width // 2, screen_height // 2,
                               text=f"Press 'r' to start '{ALL_PATTERNS[pattern_index]}' pattern",
                               font=("Helvetica", 32), fill="black")
            canvas.update()

        def on_key(event):
            nonlocal pattern_index
            if event.char == "r" and pattern_index < len(ALL_PATTERNS):
                ts_trigger = time.time_ns()
                pattern_name = ALL_PATTERNS[pattern_index]
                shared_r_triggers.append((pattern_name, ts_trigger))
                run_pattern(pattern_name)
                pattern_index += 1
                show_instruction()
            elif event.char == "s" or event.keysym == "Escape":
                stop_event.set()
                root.quit()

        root.bind("<Key>", on_key)
        show_instruction()
        root.mainloop()
        root.destroy()

    run_gui()

def _ask_user(pipe_conn, prompt, dtype):
    root = tk.Tk()
    root.withdraw()
    if dtype == "string":
        result = simpledialog.askstring("Input", prompt, parent=root)
    elif dtype == "integer":
        result = simpledialog.askinteger("Input", prompt, parent=root)
    elif dtype == "float":
        result = simpledialog.askfloat("Input", prompt, parent=root)
    else:
        result = None
    pipe_conn.send(result)
    pipe_conn.close()
    root.destroy()

def ask_user_input(prompt, dtype="string"):
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_ask_user, args=(child_conn, prompt, dtype))
    p.start()
    result = parent_conn.recv()
    p.join()
    return result

def get_session_info():
    ensure_metadata_file()
    initials = ask_user_input("Enter subject initials:", dtype="string")
    if not initials or initials.strip() == "":
        print("❌ No initials entered. Exiting.")
        exit()
    initials = initials.strip()

    with open(GLOBAL_METADATA_PATH, 'r+') as f:
        metadata = json.load(f)
        if initials not in metadata:
            print(f"🆕 New participant: {initials}")
            gender = ask_user_input("Enter gender (M/F/Other):", dtype="string")
            age = ask_user_input("Enter age:", dtype="integer")
            glasses = ask_user_input("Wearing glasses? (yes/no):", dtype="string")
            participant_info = {"gender": gender, "age": age, "glasses": glasses, "trials": {}}
            metadata[initials] = participant_info
        else:
            print(f"✅ Found existing participant: {initials}")
            participant_info = metadata[initials]

        trial_numbers = list(map(int, metadata[initials].get("trials", {}).keys()))
        next_trial = max(trial_numbers) + 1 if trial_numbers else 0

        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        participant_folder = os.path.join("sessions", initials)
        os.makedirs(participant_folder, exist_ok=True)
        trial_folder = os.path.join(participant_folder, f"{timestamp}_T{next_trial}")
        os.makedirs(trial_folder, exist_ok=True)

        f.seek(0); json.dump(metadata, f, indent=2); f.truncate()

    return trial_folder, initials, next_trial, participant_info

def update_metadata(initials, trial_number, trial_folder, shared_event_logs):
    with open(GLOBAL_METADATA_PATH, 'r+') as f:
        metadata = json.load(f)
        trial_key = str(trial_number)
        metadata[initials].setdefault("trials", {})
        metadata[initials]["trials"][trial_key] = {
            "trial_folder": trial_folder,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patterns": {}
        }
        for pattern_name, log_path in shared_event_logs.items():
            pattern_folder = os.path.dirname(log_path)
            fixation_coords_path = os.path.join(pattern_folder, "fixation_coords.csv")
            frames_dir = os.path.join(pattern_folder, "frames")
            metadata[initials]["trials"][trial_key]["patterns"][pattern_name] = {
                "event_log": log_path,
                "fixation_coords": fixation_coords_path,
                "frames_dir": frames_dir
            }
        f.seek(0); json.dump(metadata, f, indent=2); f.truncate()
    print("📝 Metadata updated.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    # 1) Session info and trial folder
    trial_folder, initials, trial_number, participant_info = get_session_info()
    video_path = os.path.join(trial_folder, "session.mp4")

    # 2) Pre-create pattern + frames dirs so they *always* exist
    precreate_pattern_dirs(trial_folder, ALL_PATTERNS)

    # 3) Shared flags/state
    recording_flag = multiprocessing.Value("b", True)
    stop_event = multiprocessing.Event()
    video_start_ns = multiprocessing.Value("q", 0)
    manager = multiprocessing.Manager()
    shared_event_logs = manager.dict()
    shared_r_triggers = manager.list()

    # shared live state (used by camera_loop for saving frames)
    shared_state = manager.dict()
    shared_state["pattern"] = None
    shared_state["fix_idx"] = -1

    # 4) Start processes
    cam_proc = multiprocessing.Process(
        target=camera_loop,
        args=(recording_flag, stop_event, video_path, video_start_ns, trial_folder, shared_state)
    )
    cam_proc.start()

    gui_proc = multiprocessing.Process(
        target=launch_gaze_target,
        args=(shared_event_logs, trial_folder, stop_event, shared_r_triggers, shared_state)
    )
    gui_proc.start()
    gui_proc.join()

    # 5) Shutdown camera
    time.sleep(0.5)
    stop_event.set()
    print("🛑 Stopping camera process...")
    cam_proc.join(timeout=5)
    if cam_proc.is_alive():
        print("⚠️ Camera process is still alive. Terminating forcefully.")
        cam_proc.terminate()
        cam_proc.join()
    print("✅ Camera process stopped.")

    # 6) Ensure video_start_ns non-zero for downstream consumers
    if video_start_ns.value == 0:
        video_start_ns.value = time.time_ns()

    # 7) Update metadata (includes frames_dir paths)
    update_metadata(initials, trial_number, trial_folder, shared_event_logs)
    print("🎉 Session completed.")

