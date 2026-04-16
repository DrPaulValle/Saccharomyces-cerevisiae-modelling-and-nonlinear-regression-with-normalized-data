"""
Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México
"""

import os
import re
import time
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# Utilities
# =============================================================================

def natural_key(name: str):
    """Sorting key for sheet names like 'E1', 'E2', ..., 'E10'."""
    m = re.match(r"^E(\d+)$", str(name).strip(), re.IGNORECASE)
    return int(m.group(1)) if m else float("inf")


def preprocess_series(arr, sigma=1.0, normalize=True):
    """Optionally smooth and/or normalize a 1D numeric series."""
    arr = np.asarray(arr, dtype=float)
    if sigma is not None and sigma > 0:
        arr = gaussian_filter1d(arr, sigma=sigma)
    if normalize:
        m = np.max(arr)
        arr = arr / (m if m != 0 else 1.0)
    return arr


def plotsys(
    t_data, w_data, x_data, y_data, z_data,
    t_plot, w_p, x_p, y_p, z_p,
    t_grid, w_e, x_e, y_e, z_e,
    title="ODE model: PINN vs Euler"
):
    """Plot a 2×2 comparison for (w, x, y, z): data vs PINN vs Euler."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    series = [
        (0, 0, "w(t): Biomass",   t_data, w_data, t_plot, w_p, t_grid, w_e, "w(t)"),
        (0, 1, "x(t): Glucose",   t_data, x_data, t_plot, x_p, t_grid, x_e, "x(t)"),
        (1, 0, "y(t): Fructose",  t_data, y_data, t_plot, y_p, t_grid, y_e, "y(t)"),
        (1, 1, "z(t): Ethanol",   t_data, z_data, t_plot, z_p, t_grid, z_e, "z(t)"),
    ]

    for r, c, ttl, td, yd, tp, yp, tg, ye, lab in series:
        ax = axes[r, c]
        ax.scatter(td, yd, s=50, color='#005F02', edgecolors="k", alpha=0.85, label=f"{lab}: data")
        ax.plot(tp, yp, lw=2, color='#134E8E', label=f"{lab}: PINN")
        ax.plot(tg, ye, "--", lw=2, color='#C00707', label=f"{lab}: Euler")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def make_table_page(
    stats, df_params, sheet_name, ic_estimates=None,
    main_fontsize=10, header_fontsize=12, ic_fontsize=10,
    ic_row_height_scale=1.3,
):
    """One-page PDF: header metrics + parameter table + optional IC table."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    R2 = stats.get("R2", np.nan)
    R2a = stats.get("R2_adj", np.nan)
    RSS = stats.get("RSS", np.nan)
    AIC = stats.get("AIC", np.nan)
    cond = stats.get("cond_JTJ", np.nan)
    used_pinv = stats.get("used_pinv", False)

    header = (
        f"{sheet_name}\n"
        f"R2={R2:.4f}, R2_adj={R2a:.4f}, RSS={RSS:.3e}, AIC={AIC:.3e}\n"
        f"cond(JTJ)={cond:.2e}, used_pinv={used_pinv}"
    )
    ax.text(0.02, 0.95, header, fontsize=header_fontsize, va="top", family="monospace")

    def fmt(v):
        try:
            v = float(v)
            return f"{v:.6e}" if np.isfinite(v) else "nan"
        except Exception:
            return str(v)

    df_disp = df_params.copy()
    for c in ["estimate", "SE", "MOE_95", "CI_low", "CI_high", "p_value"]:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].map(fmt)

    has_ic = bool(ic_estimates)
    main_bbox = [0.02, 0.30, 0.96, 0.55] if has_ic else [0.02, 0.08, 0.96, 0.80]

    t_main = ax.table(
        cellText=df_disp.values.tolist(),
        colLabels=list(df_disp.columns),
        cellLoc="center",
        colLoc="center",
        bbox=main_bbox,
    )
    t_main.auto_set_font_size(False)
    t_main.set_fontsize(main_fontsize)

    if has_ic:
        ic_items = list(ic_estimates.items())
        df_ic = pd.DataFrame({
            "IC": [k for k, _ in ic_items],
            "estimate": [fmt(v) for _, v in ic_items],
        })

        t_ic = ax.table(
            cellText=df_ic.values.tolist(),
            colLabels=list(df_ic.columns),
            cellLoc="center",
            colLoc="center",
            bbox=[0.02, 0.05, 0.45, 0.20],
        )
        t_ic.auto_set_font_size(False)
        t_ic.set_fontsize(ic_fontsize)
        t_ic.scale(1.0, ic_row_height_scale)

    return fig


# =============================================================================
# Euler solver
# =============================================================================

def euler_ODE(t_grid, w0, W0, x0, y0, z0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0):
    """
    Forward Euler integration for the 5-state model.

    Model:
        dw/dt = p1*w*(x+y)*exp(-p2*W) - p3*w*(p0+z)
        dW/dt = w
        dx/dt = -p4*x*w - p5*x
        dy/dt = -p6*y*w - p7*y
        dz/dt = p8*(p0+x+y)*w - p9*z

    Returns:
        (w, x, y, z) evaluated on t_grid (1D numpy arrays).
    """
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 2:
        raise ValueError("t_grid must have at least 2 points.")
    if np.any(np.diff(t_grid) <= 0):
        raise ValueError("t_grid must be strictly increasing.")

    N = len(t_grid)
    w = np.zeros(N); W = np.zeros(N); x = np.zeros(N); y = np.zeros(N); z = np.zeros(N)
    w[0], W[0], x[0], y[0], z[0] = map(float, (w0, W0, x0, y0, z0))

    for i in range(N - 1):
        dt = t_grid[i + 1] - t_grid[i]
        wi, Wi, xi, yi, zi = w[i], W[i], x[i], y[i], z[i]

        exp_term = np.exp(np.clip(-p2 * Wi, -50.0, 50.0))

        dw = p1 * wi * (xi + yi) * exp_term - p3 * wi * (p0 + zi)
        dW = wi
        dx = -p4 * xi * wi - p5 * xi
        dy = -p6 * yi * wi - p7 * yi
        dz = p8 * (p0 + xi + yi) * wi - p9 * zi

        w[i + 1] = wi + dt * dw
        W[i + 1] = Wi + dt * dW
        x[i + 1] = xi + dt * dx
        y[i + 1] = yi + dt * dy
        z[i + 1] = zi + dt * dz

        # optional nonnegativity clamp
        w[i + 1] = max(w[i + 1], 0.0)
        x[i + 1] = max(x[i + 1], 0.0)
        y[i + 1] = max(y[i + 1], 0.0)
        z[i + 1] = max(z[i + 1], 0.0)

    return w, x, y, z


# =============================================================================
# Parameter statistics (Euler-based, central differences)
# =============================================================================

def stats_at_optimum(t_data, t_grid, f_obs, P, w0, W0, x0, y0, z0, eps_base=1e-6):
    """Fit metrics + parameter uncertainty via central-difference Jacobian (Euler solver)."""
    t_data = np.asarray(t_data, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    f_obs = np.asarray(f_obs, dtype=float).reshape(-1, 1)
    n = f_obs.shape[0]

    P = np.asarray(P, dtype=float).ravel()
    if P.size != 10:
        raise ValueError("P must have length 10: [p1..p9,p0].")

    idx_est = [0, 1, 2, 3, 5, 7, 8]
    pnames = ["p1", "p2", "p3", "p4", "p6", "p8", "p9"]
    base = P[idx_est].copy()
    p = base.size

    def simulate_pack(P_full):
        p1_, p2_, p3_, p4_, p5_, p6_, p7_, p8_, p9_, p0_ = P_full
        w_sim, x_sim, y_sim, z_sim = euler_ODE(
            t_grid, w0, W0, x0, y0, z0,
            p1_, p2_, p3_, p4_, p5_, p6_, p7_, p8_, p9_, p0_
        )
        w_i = np.interp(t_data, t_grid, np.asarray(w_sim).ravel())
        x_i = np.interp(t_data, t_grid, np.asarray(x_sim).ravel())
        y_i = np.interp(t_data, t_grid, np.asarray(y_sim).ravel())
        z_i = np.interp(t_data, t_grid, np.asarray(z_sim).ravel())
        return np.concatenate([w_i, x_i, y_i, z_i], axis=0).reshape(-1, 1)

    f_pred = simulate_pack(P)
    resid = f_obs - f_pred
    rss = float(np.sum(resid**2))

    ss_tot = float(np.sum((f_obs - np.mean(f_obs))**2))
    R2 = np.nan if ss_tot <= 0 else 1.0 - rss / max(ss_tot, 1e-16)

    k = p
    R2_adj = np.nan if (not np.isfinite(R2) or (n - k - 1) <= 0) else \
        1.0 - (1.0 - R2) * (n - 1.0) / (n - k - 1.0)

    J = np.zeros((n, p), dtype=float)
    for j in range(p):
        h = eps_base * (1.0 + abs(base[j]))
        Pp, Pm = P.copy(), P.copy()
        Pp[idx_est[j]] += h
        Pm[idx_est[j]] -= h
        J[:, j] = ((simulate_pack(Pp) - simulate_pack(Pm)) / (2.0 * h)).ravel()

    df = n - p
    if df <= 0:
        nanv = np.full(p, np.nan)
        return dict(
            R2=R2, R2_adj=R2_adj, RSS=rss, AIC=np.nan,
            se=nanv, moe=nanv, ci_lo=nanv, ci_hi=nanv, pvals=nanv,
            names=pnames, params=base, cond_JTJ=np.nan, used_pinv=False
        )

    JTJ = J.T @ J
    try:
        cond = float(np.linalg.cond(JTJ))
    except np.linalg.LinAlgError:
        cond = float("inf")

    try:
        JTJ_inv = np.linalg.inv(JTJ)
        used_pinv = False
    except np.linalg.LinAlgError:
        JTJ_inv = np.linalg.pinv(JTJ)
        used_pinv = True

    sigma2 = rss / df
    cov = sigma2 * JTJ_inv
    se = np.sqrt(np.diag(cov).clip(min=0.0))

    tcrit = tdist.ppf(0.975, df=df)
    moe = tcrit * se
    ci_lo, ci_hi = base - moe, base + moe

    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, np.abs(base / se), np.nan)
    pvals = 2.0 * tdist.sf(tvals, df=df)

    aic = n * np.log(max(rss, 1e-16) / n) + 2 * k
    if (k > 0) and ((n / k) < 40) and ((n - k - 1) > 0):
        aic += (2 * k * (k + 1)) / (n - k - 1)

    return dict(
        R2=R2, R2_adj=R2_adj, RSS=rss, AIC=aic,
        se=se, moe=moe, ci_lo=ci_lo, ci_hi=ci_hi, pvals=pvals,
        names=pnames, params=base, cond_JTJ=cond, used_pinv=used_pinv
    )


def stats_to_table(stats):
    """Convert stats dict into a tidy parameter table."""
    names = stats["names"]
    return pd.DataFrame({
        "param": names,
        "estimate": stats.get("params", [np.nan] * len(names)),
        "SE": stats["se"],
        "MOE_95": stats["moe"],
        "CI_low": stats["ci_lo"],
        "CI_high": stats["ci_hi"],
        "p_value": stats["pvals"],
    })


# =============================================================================
# PINN model (MLP with configurable activation)
# =============================================================================

def get_activation_fn(name: str):
    """
    Return a TensorFlow activation function by name.

    Supported: tanh, relu, swish, gelu, sigmoid, softplus, elu
    """
    name = str(name).strip().lower()
    if name == "tanh":
        return tf.tanh
    if name == "relu":
        return tf.nn.relu
    if name == "swish":
        return tf.nn.swish
    if name == "gelu":
        # TF gelu exists
        return tf.nn.gelu
    if name == "sigmoid":
        return tf.nn.sigmoid
    if name == "softplus":
        return tf.nn.softplus
    if name == "elu":
        return tf.nn.elu
    raise ValueError(f"Unknown activation '{name}'. Supported: tanh, relu, swish, gelu, sigmoid, softplus, elu.")


class ODEPINN:
    def __init__(
        self,
        alpha=0.99,
        beta_ic=0.5,
        lr=1e-4,
        arch=(1, 128, 128, 128, 5),
        activation="tanh",
        seed=130425,
    ):
        """
        MLP-PINN for the 5-state ODE system with latent integral state W(t).

        Trainable:
            - NN weights/biases
            - ODE parameters: p1, p3, p4, p6, p8 (clipped positive)
            - Reparameterized: p2(theta2), p9(theta9) (bounded via sigmoid)
            - Initial condition: w0

        Fixed:
            - p5, p7, p0 constants
            - x0,y0,z0 fixed from first data point per sheet
            - W0 provided externally [W0 = 0]
        """
        self.alpha = float(alpha)
        self.beta_ic = float(beta_ic)
        self.seed = int(seed)

        self.arch = list(arch)
        if len(self.arch) < 3 or self.arch[0] != 1 or self.arch[-1] != 5:
            raise ValueError("arch must start with 1 and end with 5, e.g., (1,128,128,128,5).")

        self.act_name = str(activation)
        self.act = get_activation_fn(self.act_name)

        # fixed parameters
        self.p5 = tf.constant(np.log(2) / 840960.0, dtype=tf.float32)
        self.p7 = tf.constant(np.log(2) / 1680.0, dtype=tf.float32)
        self.p0 = tf.constant(1.0, dtype=tf.float32)

        # initial conditions
        ic_clip = lambda v: tf.clip_by_value(v, 0.0, 100)
        self.w0_var = tf.Variable(1e-1, dtype=tf.float32, constraint=ic_clip, name="w0_var")

        self.x0_fixed = None
        self.y0_fixed = None
        self.z0_fixed = None
        self.x0_tf = None
        self.y0_tf = None
        self.z0_tf = None
        self._ic_initialized = False

        # trainable positive parameters
        #init_val = 1e-1
        clip = lambda v: tf.clip_by_value(v, 1e-6, 10.0)
        self.p1 = tf.Variable(1e-2, dtype=tf.float32, constraint=clip, name="p1")
        self.p3 = tf.Variable(1e-3, dtype=tf.float32, constraint=clip, name="p3")
        self.p4 = tf.Variable(1e-1, dtype=tf.float32, constraint=clip, name="p4")
        self.p6 = tf.Variable(1e-1, dtype=tf.float32, constraint=clip, name="p6")
        self.p8 = tf.Variable(1e-2, dtype=tf.float32, constraint=clip, name="p8")

        # --- p2 bounds
        self.p2_min = tf.constant(0.0, dtype=tf.float32)
        self.p2_max = tf.constant(0.1, dtype=tf.float32)
        
        p2_min = float(self.p2_min.numpy())
        p2_max = float(self.p2_max.numpy())
        range_p2 = p2_max - p2_min
        eps2 = 1e-8 * range_p2
        
        p2_init = float(np.clip(0.05, p2_min + eps2, p2_max - eps2))
        s2 = (p2_init - p2_min) / range_p2
        theta2_init = np.log(s2 / (1.0 - s2))
        self.theta2 = tf.Variable(theta2_init, dtype=tf.float32, name="theta2")
        
        # --- p9 bounds
        self.p9_min = tf.constant(0.0, dtype=tf.float32)
        self.p9_max = tf.constant(0.1, dtype=tf.float32)
        
        p9_min = float(self.p9_min.numpy())
        p9_max = float(self.p9_max.numpy())
        range_p9 = p9_max - p9_min
        eps9 = 1e-8 * range_p9
        
        p9_init = float(np.clip(0.05, p9_min + eps9, p9_max - eps9))
        s9 = (p9_init - p9_min) / range_p9
        theta9_init = np.log(s9 / (1.0 - s9))
        self.theta9 = tf.Variable(theta9_init, dtype=tf.float32, name="theta9")

        # build MLP
        self.build_model(seed=self.seed)

        self.optimizer = tf.optimizers.Adam(learning_rate=float(lr))

        # histories
        self.loss_history = []
        self.loss_phys_hist = []
        self.loss_data_hist = []
        self.loss_ic_hist = []

        self._warned_none_derivs = False
        self._warned_none_grads = False

    def p2_value(self):
        s = tf.sigmoid(self.theta2)
        return self.p2_min + (self.p2_max - self.p2_min) * s

    def p9_value(self):
        s = tf.sigmoid(self.theta9)
        return self.p9_min + (self.p9_max - self.p9_min) * s

    def build_model(self, seed=130425):
        """Initialize manual Dense-layer weights/biases according to self.arch."""
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.weights, self.biases = [], []
        for i in range(len(self.arch) - 1):
            in_dim, out_dim = self.arch[i], self.arch[i + 1]
            W = tf.Variable(
                tf.random.truncated_normal([in_dim, out_dim], stddev=0.1, dtype=tf.float32),
                name=f"W_{i}", trainable=True
            )
            b = tf.Variable(
                tf.zeros([1, out_dim], dtype=tf.float32),
                name=f"b_{i}", trainable=True
            )
            self.weights.append(W)
            self.biases.append(b)

        self.nn_vars = self.weights + self.biases

    def neural_net(self, t):
        """
        Forward pass: t -> [w, W, x, y, z].

        Hidden activation: configurable (default tanh). Output layer: linear.
        """
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        if len(t.shape) == 1:
            t = tf.reshape(t, (-1, 1))

        X = t
        for i in range(len(self.arch) - 2):
            X = self.act(tf.matmul(X, self.weights[i]) + self.biases[i])
        return tf.matmul(X, self.weights[-1]) + self.biases[-1]

    def compute_first_derivatives(self, t, warn_none=True):
        """
        Compute states and FIRST time-derivatives.

        Returns:
            w,W,x,y,z and dw,dW,dx,dy,dz (all [N,1]).
        """
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        if len(t.shape) == 1:
            t = tf.reshape(t, (-1, 1))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            pred = self.neural_net(t)
            w, W = pred[:, 0:1], pred[:, 1:2]
            x, y, z = pred[:, 2:3], pred[:, 3:4], pred[:, 4:5]

        dw = tape.gradient(w, t)
        dW = tape.gradient(W, t)
        dx = tape.gradient(x, t)
        dy = tape.gradient(y, t)
        dz = tape.gradient(z, t)
        del tape

        if warn_none and any(g is None for g in (dw, dW, dx, dy, dz)):
            if not self._warned_none_derivs:
                names = ["dw", "dW", "dx", "dy", "dz"]
                vals = [dw, dW, dx, dy, dz]
                missing = [n for n, v in zip(names, vals) if v is None]
                print("WARNING: None derivatives:", missing)
                self._warned_none_derivs = True

        zlike = tf.zeros_like(t)
        dw = zlike if dw is None else dw
        dW = zlike if dW is None else dW
        dx = zlike if dx is None else dx
        dy = zlike if dy is None else dy
        dz = zlike if dz is None else dz

        for v in (w, W, x, y, z, dw, dW, dx, dy, dz):
            tf.ensure_shape(v, [None, 1])

        return w, W, x, y, z, dw, dW, dx, dy, dz

    def total_loss(self, t_physics, t_fit, w_fit, x_fit, y_fit, z_fit, t0, W0):
        """
        Composite loss:

            L = alpha*L_phys + (1-alpha)*L_data + beta_ic*L_ic
        """
        wS, WS, xS, yS, zS = self.scales

        w, W, x, y, z, dw, dW, dx, dy, dz = self.compute_first_derivatives(t_physics)

        p2 = self.p2_value()
        p9 = self.p9_value()
        exp_term = tf.exp(tf.clip_by_value(-p2 * W, -50.0, 50.0))

        r1 = dw - (self.p1 * w * (x + y) * exp_term - self.p3 * w * (self.p0 + z))
        r2 = dW - w
        r3 = dx - (-self.p4 * x * w - self.p5 * x)
        r4 = dy - (-self.p6 * y * w - self.p7 * y)
        r5 = dz - (self.p8 * (self.p0 + x + y) * w - p9 * z)

        L_phys = (
            tf.reduce_mean(tf.square(r1 / wS)) +
            tf.reduce_mean(tf.square(r2 / WS)) +
            tf.reduce_mean(tf.square(r3 / xS)) +
            tf.reduce_mean(tf.square(r4 / yS)) +
            tf.reduce_mean(tf.square(r5 / zS))
        )

        pred_fit = self.neural_net(t_fit)
        w_hat = pred_fit[:, 0:1]
        x_hat = pred_fit[:, 2:3]
        y_hat = pred_fit[:, 3:4]
        z_hat = pred_fit[:, 4:5]

        L_data = (
            tf.reduce_mean(tf.square((w_hat - w_fit) / wS)) +
            tf.reduce_mean(tf.square((x_hat - x_fit) / xS)) +
            tf.reduce_mean(tf.square((y_hat - y_fit) / yS)) +
            tf.reduce_mean(tf.square((z_hat - z_fit) / zS))
        )

        pred0 = self.neural_net(t0)
        w0_hat = pred0[:, 0:1]
        W0_hat = pred0[:, 1:2]
        x0_hat = pred0[:, 2:3]
        y0_hat = pred0[:, 3:4]
        z0_hat = pred0[:, 4:5]

        w0 = tf.reshape(self.w0_var, (1, 1))

        L_ic = (
            tf.reduce_mean(tf.square((w0_hat - w0) / wS)) +
            tf.reduce_mean(tf.square((W0_hat - W0) / WS)) +
            tf.reduce_mean(tf.square((x0_hat - self.x0_tf) / xS)) +
            tf.reduce_mean(tf.square((y0_hat - self.y0_tf) / yS)) +
            tf.reduce_mean(tf.square((z0_hat - self.z0_tf) / zS))
        )

        L_total = self.alpha * L_phys + (1.0 - self.alpha) * L_data + self.beta_ic * L_ic
        return L_total, L_phys, L_data, L_ic

    @tf.function
    def train_step(self, t_physics, t_fit, w_fit, x_fit, y_fit, z_fit, t0, W0):
        with tf.GradientTape() as tape:
            L_total, L_phys, L_data, L_ic = self.total_loss(
                t_physics, t_fit, w_fit, x_fit, y_fit, z_fit, t0, W0
            )

        param_vars = [self.p1, self.theta2, self.p3, self.p4, self.p6, self.p8, self.theta9]
        ic_vars = [self.w0_var]
        trainable_vars = list(self.nn_vars) + param_vars + ic_vars

        grads = tape.gradient(L_total, trainable_vars)

        if any(g is None for g in grads) and not self._warned_none_grads:
            missing = [v.name for g, v in zip(grads, trainable_vars) if g is None]
            tf.print("WARNING: None gradients for:", missing)
            self._warned_none_grads = True

        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)

        return L_total, L_phys, L_data, L_ic

    def _snapshot_state(self):
        return {
            "weights": [v.numpy().copy() for v in self.weights],
            "biases": [v.numpy().copy() for v in self.biases],
            "p1": float(self.p1.numpy()),
            "theta2": float(self.theta2.numpy()),
            "p3": float(self.p3.numpy()),
            "p4": float(self.p4.numpy()),
            "p6": float(self.p6.numpy()),
            "p8": float(self.p8.numpy()),
            "theta9": float(self.theta9.numpy()),
            "w0": float(self.w0_var.numpy()),
        }

    def _restore_state(self, st):
        for v, arr in zip(self.weights, st["weights"]):
            v.assign(arr)
        for v, arr in zip(self.biases, st["biases"]):
            v.assign(arr)
        self.p1.assign(st["p1"])
        self.theta2.assign(st["theta2"])
        self.p3.assign(st["p3"])
        self.p4.assign(st["p4"])
        self.p6.assign(st["p6"])
        self.p8.assign(st["p8"])
        self.theta9.assign(st["theta9"])
        self.w0_var.assign(st["w0"])

    def generate_data(self, t_data, w_data, x_data, y_data, z_data, n_physics, W0_value):
        t_data = np.asarray(t_data, dtype=np.float32).reshape(-1, 1)
        w_data = np.asarray(w_data, dtype=np.float32).reshape(-1, 1)
        x_data = np.asarray(x_data, dtype=np.float32).reshape(-1, 1)
        y_data = np.asarray(y_data, dtype=np.float32).reshape(-1, 1)
        z_data = np.asarray(z_data, dtype=np.float32).reshape(-1, 1)

        if t_data.shape[0] == 0:
            raise ValueError("Empty t_data.")
        if any(a.shape != t_data.shape for a in (w_data, x_data, y_data, z_data)):
            raise ValueError("t,w,x,y,z must have same length.")

        idx = np.argsort(t_data[:, 0])
        t_data, w_data, x_data, y_data, z_data = t_data[idx], w_data[idx], x_data[idx], y_data[idx], z_data[idx]

        t_min = float(t_data[0, 0])
        t_max = float(t_data[-1, 0])
        n_physics = int(n_physics)
        if n_physics < 2:
            raise ValueError("n_physics must be >= 2.")
        t_physics = np.linspace(t_min, t_max, n_physics, dtype=np.float32).reshape(-1, 1)

        t0 = t_data[0:1, :]
        W0 = np.array([[float(W0_value)]], dtype=np.float32)

        if not self._ic_initialized:
            self.w0_var.assign(float(w_data[0, 0]))
            self.x0_fixed = float(x_data[0, 0])
            self.y0_fixed = float(y_data[0, 0])
            self.z0_fixed = float(z_data[0, 0])

            self.x0_tf = tf.constant([[self.x0_fixed]], dtype=tf.float32)
            self.y0_tf = tf.constant([[self.y0_fixed]], dtype=tf.float32)
            self.z0_tf = tf.constant([[self.z0_fixed]], dtype=tf.float32)
            self._ic_initialized = True

        return t_physics, t_data, w_data, x_data, y_data, z_data, t0, W0

    def train(
        self,
        t_data, w_data, x_data, y_data, z_data,
        epochs=50000,
        verbose_every=5000,
        n_physics=600,
        W0_value=0.0,
        early_stop=True,
        window=500,
        patience=1000,
        min_rel_improve=1e-2,
    ):
        t_physics, t_data, w_data, x_data, y_data, z_data, t0, W0 = \
            self.generate_data(t_data, w_data, x_data, y_data, z_data, n_physics, W0_value)

        t_physics = tf.convert_to_tensor(t_physics, dtype=tf.float32)
        t_data = tf.convert_to_tensor(t_data, dtype=tf.float32)
        w_data = tf.convert_to_tensor(w_data, dtype=tf.float32)
        x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
        y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)
        z_data = tf.convert_to_tensor(z_data, dtype=tf.float32)
        t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
        W0 = tf.convert_to_tensor(W0, dtype=tf.float32)

        eps = tf.constant(1e-12, dtype=tf.float32)
        wS = tf.maximum(tf.reduce_max(tf.abs(w_data)), eps)
        xS = tf.maximum(tf.reduce_max(tf.abs(x_data)), eps)
        yS = tf.maximum(tf.reduce_max(tf.abs(y_data)), eps)
        zS = tf.maximum(tf.reduce_max(tf.abs(z_data)), eps)
        t_span = tf.maximum(tf.reduce_max(t_data) - tf.reduce_min(t_data), eps)
        WS = tf.maximum(wS * t_span, eps)
        self.scales = (wS, WS, xS, yS, zS)

        print(f"Loss scales | wS={float(wS.numpy()):.3e}, WS={float(WS.numpy()):.3e}, "
              f"xS={float(xS.numpy()):.3e}, yS={float(yS.numpy()):.3e}, zS={float(zS.numpy()):.3e}")
        print(f"Training 5-ODE PINN (MLP {self.arch}, activation={self.act_name}, no 2nd derivatives)")
        print(f"fixed p5={float(self.p5.numpy()):.8e}, fixed p7={float(self.p7.numpy()):.8e}, fixed p0={float(self.p0.numpy()):.8e}")
        print("-" * 80)

        if int(t_data.shape[0]) < 2:
            raise ValueError("Need at least 2 samples to use t_fit=t_data[1:].")

        t_fit = t_data[1:, :]
        w_fit = w_data[1:, :]
        x_fit = x_data[1:, :]
        y_fit = y_data[1:, :]
        z_fit = z_data[1:, :]

        best_ma = float("inf")
        best_ep = 0
        stale = 0
        ma_buffer = []
        best_state = None

        for ep in range(int(epochs)):
            L_total, L_phys, L_data, L_ic = self.train_step(
                t_physics, t_fit, w_fit, x_fit, y_fit, z_fit, t0, W0
            )

            L_total_f = float(L_total.numpy())
            L_phys_f = float(L_phys.numpy())
            L_data_f = float(L_data.numpy())
            L_ic_f = float(L_ic.numpy())

            self.loss_history.append(L_total_f)
            self.loss_phys_hist.append(L_phys_f)
            self.loss_data_hist.append(L_data_f)
            self.loss_ic_hist.append(L_ic_f)

            if early_stop:
                ma_buffer.append(L_total_f)
                if len(ma_buffer) > window:
                    ma_buffer.pop(0)

                if len(ma_buffer) == window:
                    ma = float(sum(ma_buffer) / window)
                    rel_improve = (best_ma - ma) / max(abs(best_ma), 1e-12)

                    if ma < best_ma:
                        best_ma = ma
                        best_ep = ep
                        stale = 0
                        best_state = self._snapshot_state()
                    else:
                        stale = stale + 1 if rel_improve < min_rel_improve else 0

                    if stale >= patience:
                        print(f"Early stop at epoch {ep} | best_ma={best_ma:.3e} at epoch {best_ep}")
                        break

            if (verbose_every is not None) and (verbose_every > 0) and (ep % int(verbose_every) == 0):
                contrib_phys = self.alpha * L_phys_f
                contrib_data = (1.0 - self.alpha) * L_data_f
                contrib_ic = self.beta_ic * L_ic_f
                L_check = contrib_phys + contrib_data + contrib_ic

                denom = abs(contrib_phys) + abs(contrib_data) + abs(contrib_ic) + 1e-12
                pct_phys = 100.0 * abs(contrib_phys) / denom
                pct_data = 100.0 * abs(contrib_data) / denom
                pct_ic = 100.0 * abs(contrib_ic) / denom

                print(
                    f"Epoch {ep:5d} | "
                    f"L={L_total_f:.3e} (chk={L_check:.3e}) | "
                    f"raw[phys={L_phys_f:.2e} data={L_data_f:.2e} ic={L_ic_f:.2e}] | "
                    f"share[% phys={pct_phys:4.1f} data={pct_data:4.1f} ic={pct_ic:4.1f}] | "
                    f"p1={self.p1.numpy():.2e} p2={self.p2_value().numpy():.2e} p3={self.p3.numpy():.2e} "
                    f"p4={self.p4.numpy():.2e} p6={self.p6.numpy():.2e} p8={self.p8.numpy():.2e} "
                    f"p9={self.p9_value().numpy():.2e} | w0={self.w0_var.numpy():.2e}"
                )

        if best_state is not None:
            self._restore_state(best_state)
            print(f"RESTORED best state from epoch {best_ep} | best_ma={best_ma:.3e}")

    def predict(self, t, extend_factor=1.0, dt=None):
        """
        If extend_factor > 1, interprets `t` as observed time samples and builds
        a uniform grid from min(t) to extend_factor*max(t).
    
        If extend_factor == 1, behaves like the original: predicts at the given `t`.
        """
        t = np.asarray(t, dtype=float).ravel()
    
        if extend_factor is not None and extend_factor > 1.0:
            if dt is None:
                raise ValueError("When extend_factor>1, you must provide dt.")
            t0 = float(np.min(t))
            tend = float(np.max(t))
            t_end = float(extend_factor * tend)
            n = max(2, int(np.round((t_end - t0) / dt)) + 1)
            t = np.linspace(t0, t_end, n)
    
        t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        pred = self.neural_net(tf.convert_to_tensor(t, dtype=tf.float32)).numpy()
        return t, pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4], pred[:, 4:5]

    def get_params(self):
        return (
            float(self.p1.numpy()),
            float(self.p2_value().numpy()),
            float(self.p3.numpy()),
            float(self.p4.numpy()),
            float(self.p5.numpy()),
            float(self.p6.numpy()),
            float(self.p7.numpy()),
            float(self.p8.numpy()),
            float(self.p9_value().numpy()),
            float(self.p0.numpy()),
        )

    def euler_solve(self, t_grid, w0, W0, x0, y0, z0, params=None):
        if params is None:
            params = self.get_params()
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p0 = params
        return euler_ODE(t_grid, float(w0), float(W0), float(x0), float(y0), float(z0),
                        p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)


# =============================================================================
# Batch runner
# =============================================================================

def run_all_sheets(
    excel_path,
    output_dir,
    sigma=1.0,
    normalize=True,
    epochs=50001,
    verbose_every=5000,
    n_physics=600,
    W0_value=0.0,
    dt=0.1,
    extend_factor=2.0,
    alpha=0.99,
    beta_ic=0.5,
    lr=1e-4,
    arch=(1, 128, 128, 128, 5),
    activation="tanh",
):
    """
    Run PINN + Euler + parameter uncertainty analysis for each sheet named E<number>.
    Saves per-sheet PDF+CSV and global summary.
    """
    t0_total = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    xls = pd.ExcelFile(excel_path)
    sheet_names = sorted(xls.sheet_names, key=natural_key)

    sheet_re = re.compile(r"^E\d+$", re.IGNORECASE)
    required_cols = ["t", "w", "x", "y", "z"]
    summary_rows = []

    for sheet in sheet_names:
        sheet_str = str(sheet).strip()
        if not sheet_re.match(sheet_str):
            continue

        t0_sheet = time.perf_counter()
        print(f"\n==============================\nRunning sheet: {sheet_str}\n==============================")

        df = xls.parse(sheet)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Sheet {sheet_str} missing columns: {missing}. Found: {list(df.columns)}")

        t_data = df["t"].to_numpy(dtype=float)
        if t_data.size < 2:
            raise ValueError(f"Sheet {sheet_str}: need at least 2 time points.")

        w_data = preprocess_series(df["w"].to_numpy(dtype=float), sigma, normalize)
        x_data = preprocess_series(df["x"].to_numpy(dtype=float), sigma, normalize)
        y_data = preprocess_series(df["y"].to_numpy(dtype=float), sigma, normalize)
        z_data = preprocess_series(df["z"].to_numpy(dtype=float), sigma, normalize)

        tmin = float(np.min(t_data))
        tmax = float(np.max(t_data))
        t_end = extend_factor * tmax

        model = ODEPINN(
            alpha=alpha,
            beta_ic=beta_ic,
            lr=lr,
            arch=arch,
            activation=activation,
        )
        model.train(
            t_data, w_data, x_data, y_data, z_data,
            epochs=epochs,
            verbose_every=verbose_every,
            n_physics=n_physics,
            W0_value=W0_value,
        )

        params = np.asarray(model.get_params(), dtype=float)
        w0_hat = float(model.w0_var.numpy())
        x0_hat = float(model.x0_fixed)
        y0_hat = float(model.y0_fixed)
        z0_hat = float(model.z0_fixed)

        ic_dict = {
            "w0 (estimated)": w0_hat,
            "x0 (fixed)": x0_hat,
            "y0 (fixed)": y0_hat,
            "z0 (fixed)": z0_hat,
            "W0 (fixed)": float(W0_value),
        }

        # PINN predictions
        t_plot, w_p, W_p, x_p, y_p, z_p = model.predict(t_data, extend_factor=extend_factor, dt=dt)
        
        pred_csv_path = os.path.join(output_dir, f"{sheet_str}_PINN_prediction.csv")

        df_pred = pd.DataFrame({
            "t":  np.asarray(t_plot).ravel(),
            "w":  np.asarray(w_p).ravel(),
            "W":  np.asarray(W_p).ravel(),
            "x":  np.asarray(x_p).ravel(),
            "y":  np.asarray(y_p).ravel(),
            "z":  np.asarray(z_p).ravel(),
        })
        df_pred.to_csv(pred_csv_path, index=False)
        
        print(f"Saved: {pred_csv_path}")

        # Euler simulation (extend to the same horizon as PINN)
        t_end = extend_factor * tmax
        
        n_grid = max(2, int(np.round((t_end - tmin) / (dt / 10.0))) + 1)
        t_grid = np.linspace(tmin, t_end, n_grid).astype(float)
        
        w_e, x_e, y_e, z_e = model.euler_solve(t_grid, w0_hat, W0_value, x0_hat, y0_hat, z0_hat)

        # Stats grid should stay on observed horizon only
        n_grid_stats = max(2, int(np.round((tmax - tmin) / (dt / 10.0))) + 1)
        t_grid_stats = np.linspace(tmin, tmax, n_grid_stats).astype(float)
        
        f_obs = np.asarray([w_data, x_data, y_data, z_data], dtype=np.float32).reshape(-1, 1)
        stats = stats_at_optimum(
            t_data, t_grid_stats, f_obs, P=params,
            w0=w0_hat, W0=W0_value, x0=x0_hat, y0=y0_hat, z0=z0_hat,
            eps_base=1e-6,
        )
        df_params = stats_to_table(stats)

        df_ic_excel = pd.DataFrame({
            "param":   ["w0 (estimated)", "x0 (fixed)", "y0 (fixed)", "z0 (fixed)", "W0 (fixed)"],
            "estimate":[w0_hat, x0_hat, y0_hat, z0_hat, W0_value],
            "SE":      [np.nan]*5,
            "MOE_95":  [np.nan]*5,
            "CI_low":  [np.nan]*5,
            "CI_high": [np.nan]*5,
            "p_value": [np.nan]*5,
        })
        df_stats_csv = pd.concat([df_params, df_ic_excel], ignore_index=True)

        pdf_path = os.path.join(output_dir, f"{sheet_str}_PINN_report.pdf")
        csv_table_path = os.path.join(output_dir, f"{sheet_str}_stats_table.csv")
        df_stats_csv.to_csv(csv_table_path, index=False)

        with PdfPages(pdf_path) as pdf:
            fig1 = plotsys(
                t_data, w_data, x_data, y_data, z_data,
                t_plot, w_p, x_p, y_p, z_p,
                t_grid, w_e, x_e, y_e, z_e,
                title=f"{sheet_str} | PINN vs Euler",
            )
            pdf.savefig(fig1); plt.close(fig1)

            fig2 = make_table_page(stats, df_params, sheet_str, ic_estimates=ic_dict)
            pdf.savefig(fig2); plt.close(fig2)

        elapsed_sheet = time.perf_counter() - t0_sheet
        print(f"\nTime for {sheet_str}: {elapsed_sheet:.2f} s")
        print(f"Saved: {pdf_path}")
        print(f"Saved: {csv_table_path}")

        row = {
            "sheet": sheet_str,
            "elapsed_sec": elapsed_sheet,
            "R2": stats.get("R2", np.nan),
            "R2_adj": stats.get("R2_adj", np.nan),
            "RSS": stats.get("RSS", np.nan),
            "AIC": stats.get("AIC", np.nan),
            "cond_JTJ": stats.get("cond_JTJ", np.nan),
            "used_pinv": stats.get("used_pinv", False),
            "w0_est": w0_hat,
            "x0_fixed": x0_hat,
            "y0_fixed": y0_hat,
            "z0_fixed": z0_hat,
        }
        for i, pv in enumerate(params, start=1):
            row[f"p{i}"] = float(pv)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "summary_all_sheets.csv")
    summary_df.to_csv(summary_csv, index=False)

    elapsed_total = time.perf_counter() - t0_total
    print(f"\nSaved global summary: {summary_csv}")
    print(summary_df)
    print(f"\nTOTAL elapsed time: {elapsed_total:.2f} s ({elapsed_total/60.0:.2f} min)")

    runtime_path = os.path.join(output_dir, "runtime_total.txt")
    with open(runtime_path, "w") as f:
        f.write(f"TOTAL elapsed time: {elapsed_total:.6f} seconds\n")
        f.write(f"TOTAL elapsed time: {elapsed_total/60.0:.6f} minutes\n")


if __name__ == "__main__":
    run_all_sheets(
        excel_path="data.xlsx",
        output_dir="results",
        sigma=1.0,
        normalize=False,
        epochs=100001,
        verbose_every=10000,
        n_physics=600,
        W0_value=0.0,
        dt=0.01,
        extend_factor=2.0,
        alpha=0.99,
        beta_ic=0.5,
        lr=1e-4,
        arch=(1, 128, 128, 128, 5),
        activation="tanh",
    )