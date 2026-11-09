"""
Лабораторная работа №4 (проект «со звёздочкой»).
Вариант 4: эволюция архитектуры и гиперпараметров малого классификатора
при ограниченном бюджете вычислений.

Реализованы:
    * MLP с нуля на NumPy (прямой и обратный проход, SGD + momentum);
    * генетический алгоритм для архитектуры и гиперпараметров;
    * случайный поиск при том же бюджете;
    * baseline: логистическая регрессия и «ручная» MLP;
    * 30 независимых запусков, статистика и визуализации.

Студент: Якимов Максим Андреевич
Группа: Б24-517, номер в группе 30.

Запуск:
    python main.py
"""

import os
import json
import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
BASE_SEED = 42
N_RUNS = 30
N_PER_CLASS = 250
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20
BUDGET = 240                 # число обучений на один запуск
POP_SIZE = 24
GENERATIONS = BUDGET // POP_SIZE - 1   # 9 поколений → 24*10 = 240 обучений
EPOCHS = 14
HIDDEN_MIN, HIDDEN_MAX = 8, 32
GENOME_LEN = 8
BOUNDS_LOW = np.array([1.0, 8.0, 8.0, 8.0, 0.0, -2.7, -6.0, 0.0])
BOUNDS_HIGH = np.array([3.0, 32.0, 32.0, 32.0, 1.0, -1.0, -3.0, 2.0])

ARCHIVE_LIMIT = 4000


# ---------------------------------------------------------------------------
# Данные: три концентрических класса (кольца)
# ---------------------------------------------------------------------------
def make_rings(n_per_class: int = N_PER_CLASS, noise: float = 0.28,
               seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(3):
        radius = 1.0 + 1.35 * c + rng.normal(0.0, noise, n_per_class)
        theta = rng.uniform(0.0, 2.0 * np.pi, n_per_class)
        x1 = radius * np.cos(theta)
        x2 = radius * np.sin(theta)
        X.append(np.column_stack([x1, x2]))
        y.append(np.full(n_per_class, c))
    X = np.vstack(X).astype(np.float64)
    y = np.concatenate(y).astype(int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def split_data(X: np.ndarray, y: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_train = int(TRAIN_FRAC * len(y))
    n_val = int(VAL_FRAC * len(y))
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0) + 1e-9
    Xn = (X - mean) / std
    return (Xn[tr], y[tr]), (Xn[va], y[va]), (Xn[te], y[te]), (mean, std)


# ---------------------------------------------------------------------------
# MLP с нуля
# ---------------------------------------------------------------------------
def decode_genome(g: np.ndarray) -> dict:
    g = np.clip(np.asarray(g, dtype=float), BOUNDS_LOW, BOUNDS_HIGH)
    n_layers = int(np.clip(np.round(g[0]), 1, 3))
    units = [int(np.clip(np.round(g[1 + i]), HIDDEN_MIN, HIDDEN_MAX)) for i in range(n_layers)]
    activation = "relu" if g[4] >= 0.5 else "tanh"
    lr = float(10.0 ** g[5])
    l2 = float(10.0 ** g[6])
    batch = [32, 64, 96][int(np.clip(np.round(g[7]), 0, 2))]
    return {"n_layers": n_layers, "units": units, "activation": activation,
            "lr": lr, "l2": l2, "batch": batch}


def init_params(sizes: List[int], activation: str, rng: np.random.Generator):
    params = []
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        scale = np.sqrt(2.0 / fan_in) if activation == "relu" else np.sqrt(1.0 / fan_in)
        W = rng.normal(0.0, scale, size=(fan_in, fan_out))
        b = np.zeros(fan_out)
        params.append([W, b])
    return params


def forward(X: np.ndarray, params, activation: str):
    h = X
    cache = []
    for i, (W, b) in enumerate(params):
        z = h @ W + b
        if i < len(params) - 1:
            h = np.maximum(z, 0.0) if activation == "relu" else np.tanh(z)
            cache.append((h, z))
        else:
            h = z
    return h, cache


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(X: np.ndarray, y: np.ndarray, params, activation: str) -> float:
    logits, _ = forward(X, params, activation)
    return float(np.mean(np.argmax(logits, axis=1) == y))


def train_mlp(Xtr, ytr, Xval, yval, Xte, yte, genome: np.ndarray,
              seed: int) -> dict:
    cfg = decode_genome(genome)
    rng = np.random.default_rng(seed)
    sizes = [Xtr.shape[1]] + cfg["units"] + [3]
    params = init_params(sizes, cfg["activation"], rng)
    velocity = [[np.zeros_like(W), np.zeros_like(b)] for W, b in params]

    n = len(ytr)
    n_classes = 3
    Y = np.eye(n_classes)[ytr]
    lr = cfg["lr"]
    l2 = cfg["l2"]
    batch = cfg["batch"]
    momentum = 0.9

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb, yb = Xtr[idx], Y[idx]
            logits, cache = forward(xb, params, cfg["activation"])
            probs = softmax(logits)
            dlogits = (probs - yb) / len(idx)

            for li in range(len(params) - 1, -1, -1):
                W, b = params[li]
                h_prev = xb if li == 0 else cache[li - 1][0]
                dW = h_prev.T @ dlogits + l2 * W
                db = dlogits.sum(axis=0)
                if li > 0:
                    dh = dlogits @ W.T
                    h_prev_z = cache[li - 1][1]
                    if cfg["activation"] == "relu":
                        dlogits = dh * (h_prev_z > 0)
                    else:
                        dlogits = dh * (1.0 - np.tanh(h_prev_z) ** 2)
                velocity[li][0] = momentum * velocity[li][0] - lr * dW
                velocity[li][1] = momentum * velocity[li][1] - lr * db
                params[li][0] += velocity[li][0]
                params[li][1] += velocity[li][1]

    n_params = int(sum(W.size + b.size for W, b in params))
    return {
        "cfg": cfg,
        "params": n_params,
        "val_acc": accuracy(Xval, yval, params, cfg["activation"]),
        "test_acc": accuracy(Xte, yte, params, cfg["activation"]),
        "train_acc": accuracy(Xtr, ytr, params, cfg["activation"]),
        "params_obj": params,
    }


def predict_grid(Xgrid: np.ndarray, params, activation: str) -> np.ndarray:
    logits, _ = forward(Xgrid, params, activation)
    return np.argmax(logits, axis=1)


# ---------------------------------------------------------------------------
# Генетический алгоритм
# ---------------------------------------------------------------------------
def random_genome(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(BOUNDS_LOW, BOUNDS_HIGH)


def fitness_from_result(res: dict, mode: str) -> float:
    if mode in ("compact", "ga_compact"):
        return res["val_acc"] - 5e-5 * res["params"] - 0.001 * (res["cfg"]["n_layers"] - 1)
    return res["val_acc"]


def tournament(fitness: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    idx = rng.integers(0, len(fitness), size=k)
    return int(idx[np.argmax(fitness[idx])])


def uniform_crossover(p1, p2, rng, pc=0.9):
    if rng.random() >= pc:
        return p1.copy(), p2.copy()
    mask = rng.random(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2


def gaussian_mutation(x, rng, pm=0.20, sigma=0.12):
    y = x.copy()
    mask = rng.random(len(x)) < pm
    y[mask] += rng.normal(0.0, sigma * (BOUNDS_HIGH[mask] - BOUNDS_LOW[mask]))
    return np.clip(y, BOUNDS_LOW, BOUNDS_HIGH)


def evaluate_genome(genome, data, seed, mode):
    (Xtr, ytr), (Xval, yval), (Xte, yte), _ = data
    res = train_mlp(Xtr, ytr, Xval, yval, Xte, yte, genome, seed)
    res["fitness"] = fitness_from_result(res, mode)
    res["genome"] = np.asarray(genome, dtype=float).copy()
    res["seed"] = int(seed)
    return res


def ga_run(seed: int, mode: str, data, budget: int = BUDGET) -> dict:
    rng = np.random.default_rng(seed)
    pop = np.array([random_genome(rng) for _ in range(POP_SIZE)])
    # несколько осмысленных стартовых точек
    pop[0] = [2, 32, 16, 8, 1, np.log10(0.03), -4.0, 1]
    pop[1] = [1, 32, 8, 8, 1, np.log10(0.05), -4.0, 1]
    pop[2] = [2, 16, 16, 8, 0, np.log10(0.02), -4.5, 0]
    pop[3] = [3, 32, 24, 16, 1, np.log10(0.02), -5.0, 2]

    archive = []
    eval_idx = 0
    results = []
    for g in pop:
        res = evaluate_genome(g, data, seed * 100_000 + eval_idx, mode)
        eval_idx += 1
        results.append(res)
        archive.append(res)
    fitness = np.array([r["fitness"] for r in results])

    best_idx = int(np.argmax(fitness))
    best = results[best_idx]
    history = [(0, best["fitness"], best["val_acc"], best["test_acc"], best["params"])]

    for gen in range(1, GENERATIONS + 1):
        order = np.argsort(-fitness)
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]
        new_results = [results[order[0]], results[order[1]]]
        while len(new_pop) < POP_SIZE:
            i1 = tournament(fitness, rng)
            i2 = tournament(fitness, rng)
            c1, c2 = uniform_crossover(pop[i1], pop[i2], rng)
            c1 = gaussian_mutation(c1, rng)
            c2 = gaussian_mutation(c2, rng)
            for c in (c1, c2):
                if len(new_pop) >= POP_SIZE:
                    break
                res = evaluate_genome(c, data, seed * 100_000 + eval_idx, mode)
                eval_idx += 1
                new_pop.append(c)
                new_results.append(res)
                archive.append(res)
        pop = np.array(new_pop)
        results = new_results
        fitness = np.array([r["fitness"] for r in results])
        best_idx = int(np.argmax(fitness))
        if fitness[best_idx] > best["fitness"]:
            best = results[best_idx]
        history.append((gen, best["fitness"], best["val_acc"], best["test_acc"], best["params"]))

    return {"best": best, "history": history, "archive": archive,
            "mode": mode, "seed": seed}


def random_search_run(seed: int, mode: str, data, budget: int = BUDGET) -> dict:
    rng = np.random.default_rng(seed + 777)
    archive = []
    best = None
    history = []
    for i in range(budget):
        g = random_genome(rng)
        res = evaluate_genome(g, data, seed * 100_000 + i, mode)
        archive.append(res)
        if best is None or res["fitness"] > best["fitness"]:
            best = res
        if (i + 1) % POP_SIZE == 0:
            history.append(((i + 1) // POP_SIZE, best["fitness"], best["val_acc"],
                            best["test_acc"], best["params"]))
    return {"best": best, "history": history, "archive": archive,
            "mode": mode, "seed": seed}


# ---------------------------------------------------------------------------
# Baseline-модели
# ---------------------------------------------------------------------------
def train_linear(Xtr, ytr, Xval, yval, Xte, yte, seed=0, epochs=200, lr=0.1):
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    W = rng.normal(0, 0.01, size=(d, 3))
    b = np.zeros(3)
    Y = np.eye(3)[ytr]
    for _ in range(epochs):
        logits = Xtr @ W + b
        probs = softmax(logits)
        g = (probs - Y) / n
        W -= lr * (Xtr.T @ g)
        b -= lr * g.sum(axis=0)
    def acc(X, y):
        return float(np.mean(np.argmax(X @ W + b, axis=1) == y))
    return {"train_acc": acc(Xtr, ytr), "val_acc": acc(Xval, yval),
            "test_acc": acc(Xte, yte), "params": int(W.size + b.size)}


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def plot_block_schema(path: str):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12.5, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11.6)
    ax.axis("off")

    def box(x, y, w, h, text, color="#EAF2FB", edge="#2F5D8A", fs=10):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                           linewidth=1.6, edgecolor=edge, facecolor=color)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#12263A")

    def arrow(x1, y1, x2, y2, text="", color="#2F5D8A"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                            linewidth=1.6, color=color)
        ax.add_patch(a)
        if text:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, text, fontsize=8.5, color=color)

    ax.text(7, 11.2, "Блок-схема ЛР №4: эволюция архитектуры MLP",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color="#12263A")

    cx, w = 3.7, 6.2
    steps = [
        ("Старт\n(seed = 42…71)", 10.4, "#EAF2FB", "#2F5D8A"),
        ("Инициализация популяции\nгеномов архитектуры и гиперпараметров", 9.5, "#EAF2FB", "#2F5D8A"),
        ("Декодирование генома → MLP\nи обучение на train (18 эпох)", 8.6, "#EAF2FB", "#2F5D8A"),
        ("Оценка на validation:\naccuracy − штраф за сложность", 7.7, "#FEF9E7", "#B7950B"),
        ("Турнирная селекция (k = 3),\nравномерный кроссовер", 6.8, "#EAF2FB", "#2F5D8A"),
        ("Гауссова мутация генов\nс ограничениями", 5.9, "#EAF2FB", "#2F5D8A"),
        ("Элитизм: 2 лучших генома", 5.0, "#EAF2FB", "#2F5D8A"),
        ("Оценка нового поколения\nи обновление архива", 4.1, "#EAF2FB", "#2F5D8A"),
        ("Достигнут бюджет\n480 обучений?", 3.2, "#EAF2FB", "#2F5D8A"),
        ("Выбор 3 решений разной\nсложности и качества", 2.1, "#E8F8F5", "#117A65"),
        ("Тест, сравнение с baseline,\nвизуализация", 1.0, "#E8F8F5", "#117A65"),
    ]
    for text, y, color, edge in steps:
        box(cx, y, w, 0.72, text, color=color, edge=edge)

    for i in range(len(steps) - 2):
        arrow(cx + w / 2, steps[i][1], cx + w / 2, steps[i + 1][1] + 0.72)

    arrow(cx + w / 2, 3.2, cx + w / 2, 2.82, text="да")

    loop_x = 11.9
    arrow(cx + w, 3.56, loop_x, 3.56, color="#B03A2E")
    arrow(loop_x, 3.56, loop_x, 7.16, color="#B03A2E")
    arrow(loop_x, 7.16, cx + w, 7.16, color="#B03A2E")
    ax.text(loop_x + 0.15, 5.3, "нет:\nследующее\nпоколение", fontsize=9,
            color="#B03A2E", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_convergence(all_runs: dict, path: str):
    plt.figure(figsize=(10, 5.5))
    colors = {"ga_accuracy": "#2F5D8A", "ga_compact": "#27AE60", "random_search": "#C0392B"}
    labels = {"ga_accuracy": "GA: только accuracy", "ga_compact": "GA: accuracy − сложность",
              "random_search": "Random search"}
    for mode, runs in all_runs.items():
        h = np.array([[row[2] for row in r["history"]] for r in runs])
        # выравниваем длину
        m = min(len(row) for row in h)
        h = h[:, :m]
        mean, std = h.mean(axis=0), h.std(axis=0)
        x = np.arange(m)
        plt.plot(x, mean, color=colors[mode], lw=2, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.13)
    plt.xlabel("Поколение / блок оценок")
    plt.ylabel("Лучшая validation accuracy")
    plt.title("Сходимость: среднее ± std по 30 запускам")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_box(all_runs: dict, path: str):
    labels = {"ga_accuracy": "GA accuracy", "ga_compact": "GA compact", "random_search": "Random"}
    data = [[r["best"]["test_acc"] for r in all_runs[m]] for m in all_runs]
    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(data, patch_artist=True, widths=0.55,
                     tick_labels=[labels[m] for m in all_runs])
    colors = ["#2F5D8A", "#27AE60", "#C0392B"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    plt.ylabel("Test accuracy")
    plt.title("Распределение test accuracy по 30 запускам")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_candidates(candidates: np.ndarray, path: str):
    plt.figure(figsize=(9, 5.5))
    plt.scatter(candidates[:, 1], candidates[:, 0], s=10, c="#2F5D8A", alpha=0.25)
    plt.xlabel("Число параметров модели")
    plt.ylabel("Validation accuracy")
    plt.title("Исследованные архитектуры: качество vs сложность")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_decision_boundary(model: dict, data, path: str):
    (Xtr, ytr), _, (Xte, yte), (mean, std) = data
    Xall = np.vstack([Xtr, Xte])
    x_min, x_max = Xall[:, 0].min() - 0.3, Xall[:, 0].max() + 0.3
    y_min, y_max = Xall[:, 1].min() - 0.3, Xall[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = predict_grid(grid, model["params_obj"], model["cfg"]["activation"]).reshape(xx.shape)

    plt.figure(figsize=(8, 6.5))
    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(["#D6EAF8", "#D5F5E3", "#FADBD8"])
    plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_bg, alpha=0.85)
    colors = ["#2E86C1", "#229954", "#C0392B"]
    for c in range(3):
        mask = yte == c
        plt.scatter(Xte[mask, 0], Xte[mask, 1], s=18, c=colors[c],
                    edgecolor="white", linewidth=0.3, label=f"класс {c}")
    plt.xlabel("Признак 1 (стандартизованный)")
    plt.ylabel("Признак 2 (стандартизованный)")
    plt.title("Границы решения лучшей эволюционировавшей MLP")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_architecture(model: dict, path: str):
    cfg = model["cfg"]
    sizes = [2] + cfg["units"] + [3]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    max_nodes = max(sizes)
    x_positions = np.linspace(1.0, 10.5, len(sizes))
    colors = ["#AED6F1"] + ["#A9DFBF"] * (len(sizes) - 2) + ["#F5B7B1"]
    for li, (x, n) in enumerate(zip(x_positions, sizes)):
        ys = np.linspace(3.5 - 2.2, 3.5 + 2.2, n) if n > 1 else [3.5]
        for y in ys:
            ax.add_patch(plt.Circle((x, y), 0.10, color=colors[li], ec="#2C3E50", lw=0.7))
        labels = ["Вход"] + [f"Скрытый {i + 1}" for i in range(len(sizes) - 2)] + ["Выход"]
        ax.text(x, 0.6, labels[li], ha="center", fontsize=9, color="#12263A")
        ax.text(x, 6.6, f"{n}", ha="center", fontsize=10, fontweight="bold", color="#12263A")
    for li in range(len(sizes) - 1):
        x1, x2 = x_positions[li], x_positions[li + 1]
        ys1 = np.linspace(3.5 - 2.2, 3.5 + 2.2, sizes[li]) if sizes[li] > 1 else [3.5]
        ys2 = np.linspace(3.5 - 2.2, 3.5 + 2.2, sizes[li + 1]) if sizes[li + 1] > 1 else [3.5]
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + 0.1, x2 - 0.1], [y1, y2], color="#BDC3C7", lw=0.25, alpha=0.5)
    ax.text(5.75, 6.95, f"Архитектура: {sizes}, активация = {cfg['activation']}, "
                        f"lr = {cfg['lr']:.4f}, L2 = {cfg['l2']:.1e}",
            ha="center", fontsize=10, color="#12263A")
    ax.set_title("Лучшая найденная архитектура MLP", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_dashboard(all_runs: dict, candidates: np.ndarray, model: dict, data,
                   stats: dict, path: str):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)
    fig.suptitle("ЛР №4 · Эволюция архитектуры и гиперпараметров малого классификатора",
                 fontsize=16, fontweight="bold", color="#12263A", y=0.98)

    ax1 = fig.add_subplot(gs[0, :2])
    colors = {"ga_accuracy": "#2F5D8A", "ga_compact": "#27AE60", "random_search": "#C0392B"}
    labels = {"ga_accuracy": "GA accuracy", "ga_compact": "GA compact", "random_search": "Random"}
    for mode, runs in all_runs.items():
        h = np.array([[row[2] for row in r["history"]] for r in runs])
        m = min(len(row) for row in h)
        h = h[:, :m]
        mean, std = h.mean(axis=0), h.std(axis=0)
        ax1.plot(np.arange(m), mean, color=colors[mode], lw=2, label=labels[mode])
        ax1.fill_between(np.arange(m), mean - std, mean + std, color=colors[mode], alpha=0.13)
    ax1.set_xlabel("Поколение / блок оценок")
    ax1.set_ylabel("Лучшая validation accuracy")
    ax1.set_title("Сходимость (mean ± std, 30 запусков)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 2])
    data_box = [[r["best"]["test_acc"] for r in all_runs[m]] for m in all_runs]
    bp = ax2.boxplot(data_box, patch_artist=True, widths=0.55,
                     tick_labels=[labels[m] for m in all_runs])
    for patch, c in zip(bp["boxes"], colors.values()):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("Итоговое качество")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(alpha=0.3, axis="y")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(candidates[:, 1], candidates[:, 0], s=8, c="#2F5D8A", alpha=0.25)
    ax3.set_xlabel("Параметры")
    ax3.set_ylabel("Val accuracy")
    ax3.set_title("Качество vs сложность")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    (Xtr, ytr), _, (Xte, yte), _ = data
    Xall = np.vstack([Xtr, Xte])
    x_min, x_max = Xall[:, 0].min() - 0.3, Xall[:, 0].max() + 0.3
    y_min, y_max = Xall[:, 1].min() - 0.3, Xall[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 160), np.linspace(y_min, y_max, 160))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = predict_grid(grid, model["params_obj"], model["cfg"]["activation"]).reshape(xx.shape)
    from matplotlib.colors import ListedColormap
    ax4.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5],
                 cmap=ListedColormap(["#D6EAF8", "#D5F5E3", "#FADBD8"]), alpha=0.85)
    for c, col in zip(range(3), ["#2E86C1", "#229954", "#C0392B"]):
        mask = yte == c
        ax4.scatter(Xte[mask, 0], Xte[mask, 1], s=12, c=col, edgecolor="white", linewidth=0.2)
    ax4.set_title("Граница решения")
    ax4.set_xlabel("x1")
    ax4.set_ylabel("x2")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    lines = [
        "Итоги 30 запусков",
        "",
        f"GA accuracy: {stats['ga_acc_mean']:.3f} ± {stats['ga_acc_std']:.3f}",
        f"GA compact: {stats['compact_mean']:.3f} ± {stats['compact_std']:.3f}",
        f"Random: {stats['random_mean']:.3f} ± {stats['random_std']:.3f}",
        f"Лучший test: {stats['best_test']:.3f}",
        f"Параметров: {stats['best_params']}",
        f"Архитектура: {stats['best_arch']}",
        f"p-value (GA vs random): {stats['p_value']:.2e}",
    ]
    ax5.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#EAF2FB", edgecolor="#2F5D8A"))
    ax5.set_title("Метрики", pad=10)

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Основной эксперимент
# ---------------------------------------------------------------------------
def main():
    os.makedirs("lab4_out", exist_ok=True)
    print("ЛР №4: эволюция архитектуры MLP (вариант 4)")
    print(f"Бюджет: {BUDGET} обучений на запуск, {N_RUNS} независимых запусков")

    X, y = make_rings(seed=BASE_SEED)
    data = split_data(X, y, seed=BASE_SEED)
    (Xtr, ytr), (Xval, yval), (Xte, yte), _ = data
    print(f"Данные: train={len(ytr)}, val={len(yval)}, test={len(yte)}, классов=3")

    # baseline
    baseline_lin = train_linear(Xtr, ytr, Xval, yval, Xte, yte)
    manual_genome = np.array([2, 32, 16, 8, 0, np.log10(0.03), -4.0, 1])
    baseline_mlp = train_mlp(Xtr, ytr, Xval, yval, Xte, yte, manual_genome, BASE_SEED)
    print(f"Baseline logistic regression: test={baseline_lin['test_acc']:.3f}")
    print(f"Baseline manual MLP: test={baseline_mlp['test_acc']:.3f}")

    all_runs = {"ga_accuracy": [], "ga_compact": [], "random_search": []}
    for mode in all_runs:
        print(f"\nЗапуск: {mode} ({N_RUNS} серий)...")
        for r in range(N_RUNS):
            if mode == "random_search":
                res = random_search_run(BASE_SEED + r, "accuracy", data)
            else:
                res = ga_run(BASE_SEED + r, mode, data)
            all_runs[mode].append(res)
            if (r + 1) % 10 == 0:
                print(f"  {r + 1}/{N_RUNS}: best test={res['best']['test_acc']:.3f}, "
                      f"params={res['best']['params']}")

    # CSV: результаты запусков
    rows = []
    for mode, runs in all_runs.items():
        for r, res in enumerate(runs):
            b = res["best"]
            rows.append([mode, r, b["val_acc"], b["test_acc"], b["train_acc"],
                         b["params"], b["cfg"]["n_layers"], b["cfg"]["units"],
                         b["cfg"]["activation"], b["cfg"]["lr"], b["cfg"]["l2"],
                         b["cfg"]["batch"]])
    with open("lab4_out/lab4_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "run", "val_acc", "test_acc", "train_acc", "params",
                    "n_layers", "units", "activation", "lr", "l2", "batch"])
        w.writerows(rows)

    # CSV: история сходимости
    hist_rows = []
    for mode, runs in all_runs.items():
        for r, res in enumerate(runs):
            for row in res["history"]:
                hist_rows.append([mode, r] + list(row))
    with open("lab4_out/lab4_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "run", "generation", "best_fitness", "val_acc",
                    "test_acc", "params"])
        w.writerows(hist_rows)

    # Архив кандидатов (для scatter): берём ga_accuracy и random
    candidates = []
    for mode in ["ga_accuracy", "random_search"]:
        for res in all_runs[mode]:
            for rec in res["archive"]:
                candidates.append([rec["val_acc"], rec["params"], rec["test_acc"]])
    candidates = np.array(candidates)
    np.save("lab4_out/lab4_candidates.npy", candidates)
    sample = candidates[np.random.default_rng(0).choice(
        len(candidates), size=min(ARCHIVE_LIMIT, len(candidates)), replace=False)]
    with open("lab4_out/lab4_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["val_acc", "params", "test_acc"])
        w.writerows(sample.tolist())

    # Лучшая модель по validation accuracy среди GA accuracy
    best_run = max(all_runs["ga_accuracy"], key=lambda r: r["best"]["fitness"])
    best_model = best_run["best"]
    # переобучаем на train+val для финальной оценки? Оставляем честный test:
    # модель уже обучена на train, test не использовался при отборе.
    print(f"\nЛучшая модель: val={best_model['val_acc']:.3f}, "
          f"test={best_model['test_acc']:.3f}, params={best_model['params']}")
    print(f"Архитектура: {best_model['cfg']}")

    # Три характерные модели из архива лучшего запуска
    archive = sorted(all_runs["ga_accuracy"][0]["archive"], key=lambda r: r["params"])
    # чтобы не зависеть от одного запуска, объединяем архивы всех GA-запусков
    merged = []
    for res in all_runs["ga_accuracy"]:
        merged.extend(res["archive"])
    max_val = max(r["val_acc"] for r in merged)
    compact = min([r for r in merged if r["val_acc"] >= max_val - 0.05],
                  key=lambda r: r["params"])
    accurate = max(merged, key=lambda r: (r["val_acc"], -r["params"]))
    balanced = max(merged, key=lambda r: r["val_acc"] - 1e-4 * r["params"])
    reps = [("компактная", compact), ("сбалансированная", balanced),
            ("максимальная точность", accurate)]
    with open("lab4_out/lab4_representatives.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solution", "val_acc", "test_acc", "params", "n_layers",
                    "units", "activation", "lr", "l2", "batch"])
        for name, rec in reps:
            w.writerow([name, rec["val_acc"], rec["test_acc"], rec["params"],
                        rec["cfg"]["n_layers"], rec["cfg"]["units"],
                        rec["cfg"]["activation"], rec["cfg"]["lr"],
                        rec["cfg"]["l2"], rec["cfg"]["batch"]])

    # Статистика
    ga_acc = np.array([r["best"]["test_acc"] for r in all_runs["ga_accuracy"]])
    ga_comp = np.array([r["best"]["test_acc"] for r in all_runs["ga_compact"]])
    rnd = np.array([r["best"]["test_acc"] for r in all_runs["random_search"]])
    try:
        from scipy.stats import mannwhitneyu
        p_value = float(mannwhitneyu(ga_acc, rnd, alternative="greater").pvalue)
    except Exception:
        p_value = float("nan")

    stats = {
        "variant": 4,
        "runs": N_RUNS,
        "budget_per_run": BUDGET,
        "ga_acc_mean": float(ga_acc.mean()),
        "ga_acc_std": float(ga_acc.std()),
        "compact_mean": float(ga_comp.mean()),
        "compact_std": float(ga_comp.std()),
        "random_mean": float(rnd.mean()),
        "random_std": float(rnd.std()),
        "best_test": float(best_model["test_acc"]),
        "best_val": float(best_model["val_acc"]),
        "best_params": int(best_model["params"]),
        "best_arch": [2] + best_model["cfg"]["units"] + [3],
        "p_value": p_value,
        "baseline_linear_test": baseline_lin["test_acc"],
        "baseline_manual_mlp_test": baseline_mlp["test_acc"],
        "baseline_manual_mlp_params": baseline_mlp["params"],
        "total_trainings": int(3 * N_RUNS * BUDGET),
    }
    with open("lab4_out/lab4_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Итоги ===")
    print(f"GA accuracy: {stats['ga_acc_mean']:.3f} ± {stats['ga_acc_std']:.3f}")
    print(f"GA compact:  {stats['compact_mean']:.3f} ± {stats['compact_std']:.3f}")
    print(f"Random:      {stats['random_mean']:.3f} ± {stats['random_std']:.3f}")
    print(f"Baseline LR: {stats['baseline_linear_test']:.3f}, "
          f"manual MLP: {stats['baseline_manual_mlp_test']:.3f}")
    print(f"Mann-Whitney p = {p_value:.2e}")

    print("\nПостроение графиков...")
    plot_block_schema("lab4_out/lab4_block_schema.png")
    plot_convergence(all_runs, "lab4_out/lab4_convergence.png")
    plot_box(all_runs, "lab4_out/lab4_boxplot.png")
    plot_candidates(candidates, "lab4_out/lab4_candidates.png")
    plot_decision_boundary(best_model, data, "lab4_out/lab4_decision_boundary.png")
    plot_architecture(best_model, "lab4_out/lab4_architecture.png")
    plot_dashboard(all_runs, candidates, best_model, data, stats,
                   "lab4_out/lab4_dashboard.png")
    print("Готово. Файлы в lab4_out/.")


if __name__ == "__main__":
    main()
