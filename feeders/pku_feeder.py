import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import argparse

SKELETON_EDGES = [
    # head/neck/spine/shoulder-right/arm-right chain (example)
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (10,
                                                   24),            # spine to left arm chain
    # spine to hips/legs etc.
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (6, 22),
    (0, 12), (12, 13), (13, 14), (14, 15),        # spine to right arm chain
    (0, 16), (16, 17), (17, 18), (18, 19),
]


def load_split_ids(split_path):
    with open(split_path, 'r') as f:
        lines = f.read().splitlines()

    train_ids, val_ids = [], []
    current = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "Training" in line:
            current = train_ids
        elif "Validation" in line:
            current = val_ids
        elif current is not None:
            # Remove trailing commas, split on ','
            items = [item.strip().rstrip(',')
                     for item in line.split(',') if item.strip()]
            current.extend(items)

    return set(train_ids), set(val_ids)


def get_rotation_matrix(yaw=0, pitch=0, roll=0):
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    return R_yaw @ R_pitch @ R_roll


def parse_filename(filename):
    base = os.path.basename(filename)
    m = re.match(r"(\d+)-([A-Za-z])(?:\.txt)?$", base)
    if m:
        file_id_str, view_char = m.groups()
        return int(file_id_str), view_char.upper()
    base_no_ext = os.path.splitext(base)[0]
    m2 = re.match(r"(\d+)", base_no_ext)
    if m2:
        return int(m2.group(1)), "X"
    return 0, "X"


class Feeder(Dataset):
    def __init__(self, data_dir, label_dir, split_path,
                 window_size=255, stride=50, mode='train',
                 yaw=0, pitch=0, roll=0,
                 label_agg='last',
                 seed=None, random_choose=False, random_shift=False, random_move=False,
                 normalization=False, debug=False, use_mmap=True):

        self.debug = debug

        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'"

        assert label_agg in ('none', 'mode', 'majority', 'center', 'first', 'last')
        self.label_agg = 'mode' if label_agg == 'majority' else label_agg

        self.data_dir = data_dir
        self.label_dir = label_dir
        self.window_size = window_size
        # self.stride = stride if mode == 'train' else 1
        self.stride = stride
        self.mode = mode

        train_ids, val_ids = load_split_ids(split_path)
        self.allowed_ids = train_ids if mode == 'train' else val_ids

        self.data = None
        self.label = None
        self.windows = []
        self.rotation_matrix = get_rotation_matrix(
            yaw=yaw, pitch=pitch, roll=roll)

        self.load_data()

    def load_data(self):
        print("Loading data from:", self.data_dir)
        print("Loading labels from:", self.label_dir)

        data_list = []
        label_list = []

        # list and sort data files
        file_list = sorted([f for f in os.listdir(
            self.data_dir) if f.endswith('.txt')])

        for fname in tqdm(file_list, desc=f"Loading {self.mode} samples"):
            video_name = fname.replace('.txt', '')
            file_id_num, view_char = parse_filename(fname)

            # check allowed ids: allow either numeric file_id or string video_name
            allowed = False
            if self.allowed_ids is None:
                allowed = True
            else:
                # try both integer membership and string membership
                try:
                    if (int(file_id_num) in self.allowed_ids) or (video_name in self.allowed_ids):
                        allowed = True
                except Exception:
                    if video_name in self.allowed_ids:
                        allowed = True

            if not allowed:
                continue

            data_path = os.path.join(self.data_dir, fname)
            label_path = os.path.join(self.label_dir, f"{video_name}.pkl")

            # load data
            data = np.loadtxt(data_path)  # shape (Ti, 150)

            # load labels: support both new tuple format and old plain array
            try:
                with open(label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(
                        f, encoding='latin1')

            self.label = np.asarray(self.label, dtype=np.int64).reshape(-1)

            if data.shape[0] != self.label.shape[0]:
                raise ValueError(
                    f"Frame count mismatch for {fname}: data frames = {data.shape[0]}, label frames = {self.label.shape[0]}"
                )

            data_list.append(data)           # append (Ti, 150)
            label_list.append(self.label)     # append (Ti,)

        if len(data_list) == 0:
            raise RuntimeError(
                "No files loaded — check data_dir, label_dir and split ids.")

        # concatenate all sequences (time-concatenation)
        self.data = np.concatenate(data_list)      # shape [T_total, 150]
        self.label = np.concatenate(label_list)  # shape [T_total,]

        if self.debug:
            self.label = self.label[0:1000]
            self.data = self.data[0:1000]
            self.sample_name = self.sample_name[0:1000]

        total_frames = len(self.data)
        self.windows = [
            start for start in range(0, total_frames - self.window_size + 1, self.stride)
        ]

        print(f"Total concatenated frames: {total_frames}")
        print(f"Total windows: {len(self.windows)}")

        self.window_labels = []
        for start in self.windows:
            label_window = self.label[start:start + self.window_size]
            if self.label_agg == 'none':
                out_label = label_window.astype(np.int64)
            elif self.label_agg == 'first':
                out_label = np.int64(label_window[0])
            elif self.label_agg == 'center':
                out_label = np.int64(label_window[len(label_window) // 2])
            elif self.label_agg == 'last':
                out_label = np.int64(label_window[-1])
            elif self.label_agg == 'mode':
                if label_window.size == 0:
                    out_label = np.int64(0)
                else:
                    counts = np.bincount(label_window.astype(np.int64))
                    out_label = np.int64(np.argmax(counts))
            else:
                out_label = label_window.astype(np.int64)

            self.window_labels.append(out_label)

        self.window_labels = np.array(self.window_labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        end = start + self.window_size

        data_window = self.data[start:end]      # (window_size, 150)
        label_window = self.label[start:end]   # (window_size,)
        out_label = self.window_labels[idx]

        last_class = label_window[-1]
        distance = 0
        for i in range(len(label_window) - 2, -1, -1):
            if label_window[i] == last_class:
                distance += 1
            else:
                break

        # Reshape: (T, 150) → (T, 25, 3, 2)
        data_window = data_window.reshape(-1, 2, 25, 3)  # (T, 2, 25, 3)
        data_window = data_window @ self.rotation_matrix
        data_window = data_window.reshape(self.window_size, 150)
        # data_window = np.transpose(data_window, (3, 0, 2, 1))  # (3, T, 25, 2)

        return data_window.astype(np.float32), out_label, distance, idx

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:]
                     for i, l in enumerate(self.window_labels)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


# def make_3d_anim(frames, edges=None, show_second=True,
#                  elev=20, azim=45, interval=40,
#                  save_as=None, dpi=100):
#     """
#     frames: (T, M, V, 3)
#     edges: list of (u,v) pairs (0-based)
#     show_second: whether to plot second person if present
#     interval: ms between frames
#     save_as: str or None, if path ends with .mp4 or .gif will attempt to save
#     """

#     # T, M, V, C = frames.shape
#     # frames = frames.transpose(1, 3, 2, 0)
#     frames = frames.reshape(-1, 2, 25, 3)  # (T, 2, 25, 3)
#     print(frames)
#     # frames = frames.transpose(3, 0, 2, 1)  # (3, T, 25, 2)
#     T, M, V, C = frames.shape
#     # detect if second person exists anywhere
#     has_second = np.any(np.abs(frames[:,1]).sum(axis=(1,2)) > 1e-6) if M > 1 else False

#     fig = plt.figure(figsize=(6,6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.view_init(elev=elev, azim=azim)

#     # compute global bounds for axes
#     nonzeros = frames.reshape(-1,3)[np.abs(frames.reshape(-1,3)).sum(axis=1) > 0]
#     if nonzeros.size == 0:
#         raise ValueError("All frames are zero!")
#     mins = nonzeros.min(axis=0)
#     maxs = nonzeros.max(axis=0)
#     rng = max(maxs - mins)
#     mid = (maxs + mins) / 2.0

#     ax.set_xlim(mid[0] - rng*0.6, mid[0] + rng*0.6)
#     ax.set_ylim(mid[1] - rng*0.6, mid[1] + rng*0.6)
#     ax.set_zlim(mid[2] - rng*0.6, mid[2] + rng*0.6)
#     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

#     ax.set_title("PKU-MMD Skeleton Visualization")

#     def update(frame_idx):
#         ax.cla()  # safer clear of axes before drawing
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_xlim(mid[0] - rng*0.6, mid[0] + rng*0.6)
#         ax.set_ylim(mid[1] - rng*0.6, mid[1] + rng*0.6)
#         ax.set_zlim(mid[2] - rng*0.6, mid[2] + rng*0.6)
#         ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#         ax.set_title(f"Frame {frame_idx+1}/{T}")

#         f = frames[frame_idx]
#         artists = []
#         # Plot first person
#         p0 = f[0]
#         # skip person if all zeros
#         if not np.allclose(p0, 0.0):
#             xs, ys, zs = p0[:,0], p0[:,1], p0[:,2]
#             ax.scatter(xs, ys, zs, s=30, c='b', label='P0')
#             if edges:
#                 for (u,v) in edges:
#                     # safety check for indices
#                     if u >= V or v >= V:
#                         continue
#                     l, = ax.plot([p0[u,0], p0[v,0]],
#                             [p0[u,1], p0[v,1]],
#                             [p0[u,2], p0[v,2]], c='b', lw=2)
#                     artists.append(l)

#         # Plot second person if exists
#         if has_second and show_second:
#             p1 = f[1]
#             if not np.allclose(p1, 0.0):
#                 xs2, ys2, zs2 = p1[:,0], p1[:,1], p1[:,2]
#                 ax.scatter(xs2, ys2, zs2, s=30, c='r', label='P1')
#                 if edges:
#                     for (u,v) in edges:
#                         if u >= V or v >= V:
#                             continue
#                         l2, = ax.plot([p1[u,0], p1[v,0]],
#                                 [p1[u,1], p1[v,1]],
#                                 [p1[u,2], p1[v,2]], c='r', lw=2)
#                         artists.append(l2)

#         return artists

#     anim = animation.FuncAnimation(fig, update, frames=range(T), interval=interval, blit=False)

#     if save_as:
#         save_as = str(save_as)
#         ext = os.path.splitext(save_as)[1].lower()
#         if ext == '.mp4':
#             Writer = animation.writers['ffmpeg']
#             writer = Writer(fps=1000.0/interval, metadata=dict(artist='Me'), bitrate=1800)
#             anim.save(save_as, writer=writer, dpi=dpi)
#         elif ext == '.gif':
#             anim.save(save_as, writer='pillow', fps=1000.0/interval, dpi=dpi)
#         else:
#             print("Unknown save extension, skipping save.")
#     else:
#         plt.show()

#     return anim

# def make_3d_anim(frames, edges=None, show_second=True,
#                  elev=20, azim=45, interval=40,
#                  save_as=None, dpi=100):
#     frames = frames.reshape(-1, 2, 25, 3)  # (T, 2, 25, 3)
#     print('frames after reshape in anim:', frames.shape)
#     T, M, V, C = frames.shape

#     has_second = np.any(np.abs(frames[:,1]).sum(axis=(1,2)) > 1e-6) if M > 1 else False

#     fig = plt.figure(figsize=(6,6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.view_init(elev=elev, azim=azim)

#     # compute global bounds
#     nonzeros = frames.reshape(-1,3)[np.abs(frames.reshape(-1,3)).sum(axis=1) > 0]
#     if nonzeros.size == 0:
#         raise ValueError("All frames are zero!")
#     mins = nonzeros.min(axis=0)
#     maxs = nonzeros.max(axis=0)
#     rng = max(maxs - mins)
#     mid = (maxs + mins) / 2.0

#     ax.set_xlim(mid[0] - rng*0.6, mid[0] + rng*0.6)
#     ax.set_ylim(mid[1] - rng*0.6, mid[1] + rng*0.6)
#     ax.set_zlim(mid[2] - rng*0.6, mid[2] + rng*0.6)
#     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#     ax.set_title("PKU-MMD Skeleton Visualization")

#     # --- Prepare persistent artists ---
#     scatters = [ax.scatter([], [], [], s=30, c='b'),  # P0
#                 ax.scatter([], [], [], s=30, c='r')]  # P1
#     lines = []
#     if edges:
#         for p in range(2):
#             lines.append([ax.plot([], [], [], c='b' if p==0 else 'r', lw=2)[0] for _ in edges])

#     def update(frame_idx):
#         f = frames[frame_idx]

#         # Update first person scatter
#         p0 = f[0]
#         if not np.allclose(p0, 0.0):
#             scatters[0]._offsets3d = (p0[:,0], p0[:,1], p0[:,2])
#             if edges:
#                 for i, (u,v) in enumerate(edges):
#                     lines[0][i].set_data([p0[u,0], p0[v,0]], [p0[u,1], p0[v,1]])
#                     lines[0][i].set_3d_properties([p0[u,2], p0[v,2]])
#         else:
#             scatters[0]._offsets3d = ([], [], [])

#         # Update second person
#         if has_second and show_second:
#             p1 = f[1]
#             if not np.allclose(p1, 0.0):
#                 scatters[1]._offsets3d = (p1[:,0], p1[:,1], p1[:,2])
#                 if edges:
#                     for i, (u,v) in enumerate(edges):
#                         lines[1][i].set_data([p1[u,0], p1[v,0]], [p1[u,1], p1[v,1]])
#                         lines[1][i].set_3d_properties([p1[u,2], p1[v,2]])
#             else:
#                 scatters[1]._offsets3d = ([], [], [])

#         return scatters + sum(lines, [])

#     anim = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

#     if save_as:
#         save_as = str(save_as)
#         ext = os.path.splitext(save_as)[1].lower()
#         fps = 1000.0/interval
#         if ext == '.mp4':
#             Writer = animation.writers['ffmpeg']
#             writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
#             anim.save(save_as, writer=writer, dpi=dpi)
#         elif ext == '.gif':
#             anim.save(save_as, writer='pillow', fps=fps, dpi=dpi)
#         else:
#             print("Unknown save extension, skipping save.")
#     else:
#         plt.show()

#     return anim

def make_3d_anim(frames, edges, save_as='anim.gif', interval=50):
    """
    frames: np.array of shape (T, M, V, 3)
    edges: list of tuples (start_joint, end_joint)
    save_as: filename to save animation
    interval: time between frames in ms
    """

    T, M, V, _ = frames.shape

    # Replace all-zero joints with np.nan to avoid plotting errors
    frames_nan = frames.copy()
    zero_mask = np.all(frames_nan == 0, axis=-1)  # shape: (T, M, V)
    frames_nan[zero_mask] = np.nan

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits based on the data
    all_coords = frames_nan.reshape(-1, 3)
    ax.set_xlim(np.nanmin(all_coords[:, 0]), np.nanmax(all_coords[:, 0]))
    ax.set_ylim(np.nanmin(all_coords[:, 1]), np.nanmax(all_coords[:, 1]))
    ax.set_zlim(np.nanmin(all_coords[:, 2]), np.nanmax(all_coords[:, 2]))

    # Initialize Line3D objects for each edge
    lines = []
    for m in range(M):
        for start, end in edges:
            line, = ax.plot([0, 0], [0, 0], [0, 0], 'o-', lw=2)
            lines.append(line)

    def update(frame_idx):
        line_idx = 0
        for m in range(M):
            for start, end in edges:
                xs = [frames_nan[frame_idx, m, start, 0],
                      frames_nan[frame_idx, m, end, 0]]
                ys = [frames_nan[frame_idx, m, start, 1],
                      frames_nan[frame_idx, m, end, 1]]
                zs = [frames_nan[frame_idx, m, start, 2],
                      frames_nan[frame_idx, m, end, 2]]

                lines[line_idx].set_data(xs, ys)
                lines[line_idx].set_3d_properties(zs)
                line_idx += 1
        return lines

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    anim.save(save_as, writer='pillow', fps=1000/interval, dpi=80)
    plt.close(fig)
    print(f"Animation saved to {save_as}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default=None,
                        help='save animation to file (mp4 or gif)')
    parser.add_argument('--interval', type=int, default=40,
                        help='ms between frames')
    args = parser.parse_args()

    DATA_DIR = "/home/dsp520/Annisa/Online-AAGCN/data/pkummd/Data/PKU_Skeleton_Renew"
    LABEL_DIR = "/home/dsp520/Annisa/Online-AAGCN/data/pkummd/Label"
    SPLIT_PATH = "/home/dsp520/Annisa/Online-AAGCN/data/pkummd/Split/cross-subject.txt"

    train_dataset = Feeder(
        data_dir=DATA_DIR,
        label_dir=LABEL_DIR,
        split_path=SPLIT_PATH,
        window_size=255,
        stride=1,
        mode='train',
        # yaw=90,
        pitch=-90
    )

    val_dataset = Feeder(
        data_dir=DATA_DIR,
        label_dir=LABEL_DIR,
        split_path=SPLIT_PATH,
        window_size=255,
        stride=1,
        mode='val'
    )

    def my_collate(batch):
        data = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        label = torch.tensor([b[1] for b in batch], dtype=torch.long)
        distance = torch.tensor([b[2] for b in batch],
                                dtype=torch.long)  # convert int to tensor
        idx = torch.tensor([b[3] for b in batch], dtype=torch.long)
        return data, label, distance, idx

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, collate_fn=my_collate)

    print("\n--- Testing Train Loader ---")
    for i, (data, label, distance, idx) in enumerate(train_loader):
        print(f"Batch {i} | Data: {data.shape} | Label: {label.shape}")
        if i == 1:
            break

    print("\n--- Testing Val Loader ---")
    for i, (data, label, distance, idx) in enumerate(val_loader):
        print(f"Sample {i} | Data: {data.shape} | Label: {label.shape}")
        if i == 1:
            break

    # testing data visualization

    print('example one train_dataset: ', len(train_dataset))
    sample_data, sample_label, distance, idx = train_dataset[1002]
    print('sample_data before anim:', sample_data.shape)
    print('sample_label before anim:', sample_label.shape)
    print(sample_label)
    print('distance: ', distance)

    # edges = SKELETON_EDGES
    # frames = sample_data.reshape(-1, 2, 25, 3)
    # make_3d_anim(frames, edges=edges, save_as=args.save, interval=args.interval)


if __name__ == "__main__":
    main()
