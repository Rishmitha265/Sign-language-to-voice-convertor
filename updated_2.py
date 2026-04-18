

import os
import cv2
import time
import json
import threading
import numpy as np
import mediapipe as mp
from tkinter import Tk, Frame, Button, Label, Entry, StringVar, messagebox, simpledialog
from tkinter import filedialog
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

# -------------------- Settings --------------------
DATA_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# MediaPipe setup (global to reuse)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
HAND_PROC = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)

# -------------------- Utilities --------------------

def normalize_landmarks(landmarks, frame_w, frame_h):
    """Convert mediapipe normalized landmarks to centered, scale-normalized vector (63,)
    """
    pts = np.array([[lm.x * frame_w, lm.y * frame_h, lm.z * frame_w] for lm in landmarks], dtype=np.float32)
    center = pts.mean(axis=0)
    pts -= center
    max_dist = np.max(np.linalg.norm(pts, axis=1)) + 1e-8
    pts /= max_dist
    return pts.flatten()


def ensure_gesture_dir(gesture):
    d = os.path.join(DATA_DIR, gesture)
    os.makedirs(d, exist_ok=True)
    return d

# -------------------- Data Collection (single-frame only) --------------------

def collect_single_frame(gesture_name, samples=200, show_window=True):
    d = ensure_gesture_dir(gesture_name)
    cap = cv2.VideoCapture(0)
    count = len([f for f in os.listdir(d) if f.endswith('.npy') and not f.startswith('seq_')])
    print(f"Collecting single-frame for '{gesture_name}'. Existing samples={count}")

    started = False  # flag to start auto-saving after pressing 's'

    try:
        while count < samples:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = HAND_PROC.process(frame_rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                vec = normalize_landmarks(lm, w, h)
                mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                if not started:
                    cv2.putText(frame, "Press 's' to START saving", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                else:
                    fname = os.path.join(d, f"{count:04d}.npy")
                    np.save(fname, vec)
                    count += 1
                    cv2.putText(frame, f"Saving... {count}/{samples}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "No hand detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if show_window:
                cv2.imshow(f"Collect - {gesture_name}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    started = True   # begin auto saving
                    print("Started saving samples...")
                elif key == ord('q'):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
    print("Collection finished.")

# -------------------- Models (landmark single-frame) --------------------

def build_landmark_model(input_dim=63, num_classes=5):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------- Dataset loading & training (single-frame) --------------------

def load_single_frame_dataset():
    labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    X, y = [], []
    label_map = {name:i for i,name in enumerate(labels)}
    for name in labels:
        d = os.path.join(DATA_DIR, name)
        for f in os.listdir(d):
            if f.endswith('.npy') and not f.startswith('seq_'):
                arr = np.load(os.path.join(d,f))
                X.append(arr.astype(np.float32))
                y.append(label_map[name])
    if not X:
        return None, None, {}
    return np.stack(X), np.array(y, dtype=np.int32), label_map


def augment_noise(x, sigma=0.02):
    return x + np.random.normal(0, sigma, size=x.shape)


def train_landmark_model(epochs=80, batch=32):
    X, y, label_map = load_single_frame_dataset()
    if X is None:
        messagebox.showerror("Error", "No single-frame data found. Collect samples first.")
        return
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    model = build_landmark_model(input_dim=X.shape[1], num_classes=len(label_map))
    ckpt_path = os.path.join(MODEL_DIR, 'best_landmark.h5')
    ckpt = ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max')
    rlrop = ReduceLROnPlateau(monitor='val_loss', patience=6)
    early = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

    def gen(Xs, ys, batch=32):
        n = Xs.shape[0]
        while True:
            idx = np.random.choice(n, batch)
            batch_x = Xs[idx].copy()
            for i in range(batch):
                if np.random.rand() < 0.6:
                    batch_x[i] = augment_noise(batch_x[i], sigma=0.03)
            yield batch_x, ys[idx]

    steps = max(1, X_train.shape[0] // batch)
    model.fit(gen(X_train, y_train), steps_per_epoch=steps, epochs=epochs,
              validation_data=(X_val, y_val), callbacks=[ckpt, rlrop, early])
    final_path = os.path.join(MODEL_DIR, 'final_landmark.h5')
    model.save(final_path)
    with open(os.path.join(MODEL_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    messagebox.showinfo("Training", f"Training complete. Model saved to {final_path}")

# -------------------- Live Prediction (landmark) --------------------

def live_predict_landmark(model_path=None):
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'best_landmark.h5')
    label_map_path = os.path.join(MODEL_DIR, 'label_map.json')
    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        messagebox.showerror("Error", "Model or label_map not found. Train first.")
        return
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    inv_map = {int(v):k for k,v in label_map.items()}
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            h,w,_ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = HAND_PROC.process(frame_rgb)
            text = "No hand"
            if res.multi_hand_landmarks:
                vec = normalize_landmarks(res.multi_hand_landmarks[0].landmark, w, h)
                pred = model.predict(vec.reshape(1,-1), verbose=0)
                idx = int(np.argmax(pred))
                conf = float(pred[0, idx])
                text = f"{inv_map[idx]} ({conf:.2f})"
                mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Live Predict - Landmark", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# -------------------- GUI --------------------

class App:
    def __init__(self, root):
        self.root = root
        root.title("Hand Gesture Toolkit")
        frm = Frame(root, padx=12, pady=12)
        frm.pack()

        Button(frm, text="Add Gesture", width=30, command=self.add_gesture).pack(pady=6)
        Button(frm, text="Collect Single-Frame Samples", width=30, command=self.collect_single_ui).pack(pady=6)
        Button(frm, text="Train Landmark Model", width=30, command=self.train_landmark_thread).pack(pady=6)
        Button(frm, text="Live Predict (Landmark)", width=30, command=self.live_landmark_thread).pack(pady=6)
        Button(frm, text="Export / Select Model File", width=30, command=self.select_model_file).pack(pady=6)
        Button(frm, text="Quit", width=30, command=root.quit).pack(pady=6)
        self.selected_model = None

    def add_gesture(self):
        name = simpledialog.askstring("Gesture name", "Enter a gesture name (no spaces):")
        if not name: return
        ensure_gesture_dir(name)
        messagebox.showinfo("Created", f"Created dataset/{name}")

    def collect_single_ui(self):
        name = simpledialog.askstring("Collect Single", "Gesture name to collect:")
        if not name: return
        samples = simpledialog.askinteger("Samples", "How many samples?", initialvalue=200, minvalue=10, maxvalue=5000)
        t = threading.Thread(target=collect_single_frame, args=(name, samples), daemon=True)
        t.start()

    def train_landmark_thread(self):
        t = threading.Thread(target=train_landmark_model, daemon=True)
        t.start()

    def live_landmark_thread(self):
        model_file = self.selected_model if self.selected_model else None
        t = threading.Thread(target=live_predict_landmark, args=(model_file,), daemon=True)
        t.start()

    def select_model_file(self):
        f = filedialog.askopenfilename(title='Select model (.h5)', filetypes=[('H5 models', '*.h5'), ('All files','*.*')])
        if f:
            self.selected_model = f
            messagebox.showinfo("Selected", f"Selected model: {f}")


if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()
