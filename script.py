import numpy as np
import matplotlib.pyplot as plt

# P = (np.random.random(size=(100, 100)) * 2) - 1
# P += 1
# P /= 2
# P = np.round(255 * P)

# print(P)
# print(P.min(), P.max())
# plt.imshow(P, cmap=plt.cm.binary)
# plt.show()

import numpy as np
from noise import pnoise3
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 200 # size of the 3D matrix
timeSteps = 100
theta = 30

# Create a (N, N, N) shaped numpy array filled with Perlin noise values
dataA = np.array([[[pnoise3(i / theta, j / theta, k / theta, octaves=3, persistence=0.3, lacunarity=1) for i in range(timeSteps)] for j in range(N)] for k in range(N)])
dataB = np.array([[[pnoise3(i / theta, j / theta, k / theta, octaves=3, persistence=0.3, lacunarity=1) for i in range(timeSteps)] for j in range(N)] for k in range(N)])
dataB = np.flip(dataB, axis=0)
dataB = np.flip(dataB, axis=1)
print(f'Created data of shape: {dataA.shape}')
print(f'Data range: {dataA.min(), dataA.max()}')
print(f'Created data of shape: {dataB.shape}')
print(f'Data range: {dataB.min(), dataB.max()}')

fig, axs = plt.subplots(1)
# fig, axs = plt.subplots(2)
plt.margins()

# Function to update the plot for each frame of the animation
def update(num):
    # # As perlin frame animation
    # for ax, data in zip(axs, [dataA, dataB]):
    #     ax.clear()
    #     ax.set_title(f'Perlin noise slice at z = {num}')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.imshow(data[:, :, num], cmap=plt.cm.binary)

    # As arrows
    X, Y = np.meshgrid(np.linspace(0, 100, N),
                       np.linspace(0, 100, N))
    U = dataA[:, :, num]
    V = dataB[:, :, num]

    # print(U[N//2, N//2], V[N//2, N//2])

    axs.clear()
    axs.set_title(f'Frame: {num}')
    axs.quiver(
        X, Y, U, V,
        angles='uv',
        headwidth=0.50,
        headlength=0.50,
        # scale=0.50
    )


# Create the animation with 100 frames
ani = FuncAnimation(fig, update, frames=range(timeSteps), repeat=True)
# update(0)

plt.show()
