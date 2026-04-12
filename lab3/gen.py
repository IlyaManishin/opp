import random

N = 1000
K = 2000
M = 1500
FILENAME = "data.txt"

def generate():
    with open(FILENAME, "w") as f:
        f.write(f"{N} {K} {M}\n")

        for _ in range(N * K):
            f.write(f"{random.uniform(-10, 10):.6f} ")

        for _ in range(K * M):
            f.write(f"{random.uniform(-10, 10):.6f} ")

if __name__ == "__main__":
    generate()