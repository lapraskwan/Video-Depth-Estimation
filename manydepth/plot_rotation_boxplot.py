import torch
import matplotlib.pyplot as plt

kitti_intrinsic = torch.load("plot/kitti_intrinsic/rotation.pt")
nyu50k_intrinsic = torch.load("plot/nyu50k_intrinsic/rotation.pt")
nyu50k_from_kitti_intrinsic = torch.load("plot/nyu50k_from_kitti_intrinsic/rotation.pt")

rx = (kitti_intrinsic["rx"], nyu50k_intrinsic["rx"], nyu50k_from_kitti_intrinsic["rx"])
ry = (kitti_intrinsic["ry"], nyu50k_intrinsic["ry"], nyu50k_from_kitti_intrinsic["ry"])
rz = (kitti_intrinsic["rz"], nyu50k_intrinsic["rz"], nyu50k_from_kitti_intrinsic["rz"])

fig, ax = plt.subplots()
ax.boxplot(rx, labels=["kitti", "nyu50k", "kitti2nyu"], showfliers=False)
ax.set_ylabel("rx")
fig.savefig("plot/rx_boxplot.png")

fig, ax = plt.subplots()
ax.boxplot(ry, labels=["kitti", "nyu50k", "kitti2nyu"], showfliers=False)
ax.set_ylabel("ry")
fig.savefig("plot/ry_boxplot.png")

fig, ax = plt.subplots()
ax.boxplot(rz, labels=["kitti", "nyu50k", "kitti2nyu"], showfliers=False)
ax.set_ylabel("rz")
fig.savefig("plot/rz_boxplot.png")