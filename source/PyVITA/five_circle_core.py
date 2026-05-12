"""Minimal 5-circle tunnel cross-section fitting core for PyVITA."""

import numpy as np
def wrap_pi(x: np.ndarray) -> np.ndarray:
    """[Overview] Wrap angle values to (-pi, pi]."""
    return np.arctan2(np.sin(x), np.cos(x))

def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    """[Overview] Return 2D cross-product scalar a.x*b.y - a.y*b.x."""
    return float(a[0] * b[1] - a[1] * b[0])

def clip_uv_top_percent_by_z_and_plot(
    uv: np.ndarray,
    top_range: float,
    z_col: int = 1,
):
    """
    Keep top `top_range` of z-range from uv.

    Args:
        uv: (N, 2) projected points.
        top_range: percentage of z-range to keep from high-z side (0 < p <= 100).
        z_col: uv column index used as z axis (0 or 1).

    Returns:
        {
            "uv_clipped": (M, 2),
            "mask_kept": (N,) bool,
            "z_threshold": float,
        }
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if not (0.0 < float(top_range) <= 100.0):
        raise ValueError("top_range must satisfy 0 < top_range <= 100")
    if z_col not in (0, 1):
        raise ValueError("z_col must be 0 or 1")

    z_vals = uv[:, z_col]
    keep_ratio = float(top_range) / 100.0
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    z_threshold = z_min + (1.0 - keep_ratio) * (z_max - z_min)

    mask_kept = z_vals >= z_threshold
    uv_clipped = uv[mask_kept]

    return {
        "uv_clipped": uv_clipped,
        "mask_kept": mask_kept,
        "z_threshold": z_threshold,
    }

def _circle_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """Return (center, radius) from 3 points, or (None, None) if degenerate."""
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if np.isclose(d, 0.0):
        return None, None

    x1s, y1s = x1 * x1, y1 * y1
    x2s, y2s = x2 * x2, y2 * y2
    x3s, y3s = x3 * x3, y3 * y3

    ux = ((x1s + y1s) * (y2 - y3) + (x2s + y2s) * (y3 - y1) + (x3s + y3s) * (y1 - y2)) / d
    uy = ((x1s + y1s) * (x3 - x2) + (x2s + y2s) * (x1 - x3) + (x3s + y3s) * (x2 - x1)) / d

    center = np.array([ux, uy], dtype=np.float64)
    radius = float(np.linalg.norm(p1 - center))
    if not np.isfinite(radius) or radius <= 0.0:
        return None, None
    return center, radius

def _refit_circle_least_squares(uv: np.ndarray):
    """Algebraic least-squares circle fit."""
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    x = uv[:, 0]
    y = uv[:, 1]
    A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0.0:
        raise ValueError("Least-squares refit produced invalid radius")
    center = np.array([cx, cy], dtype=np.float64)
    radius = float(np.sqrt(r2))
    return center, radius

def fit_circle_ransac(
    uv: np.ndarray,
    residual_threshold: float,
    max_trials: int = 2000,
    min_inliers: int | None = None,
    random_seed: int | None = 0,
    refit: bool = True,
) -> dict[str, np.ndarray | float]:
    """
    Fit a circle to 2D points with RANSAC and return inliers.

    Args:
        uv: (N,2) points.
        residual_threshold: inlier condition |dist_to_center - radius| <= threshold.
        max_trials: number of random 3-point hypotheses.
        min_inliers: optional minimum required inlier count.
        random_seed: random seed for reproducibility.
        refit: if True, refit circle using all inliers by least squares.

    Returns:
        {
            "center": (2,),
            "radius": float,
            "inlier_mask": (N,) bool,
            "inliers": (M,2),
            "residuals": (N,),
            "num_inliers": int,
            "inlier_ratio": float,
        }
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if uv.shape[0] < 3:
        raise ValueError("Need at least 3 points for circle fitting")
    if residual_threshold <= 0:
        raise ValueError("residual_threshold must be > 0")
    if max_trials < 1:
        raise ValueError("max_trials must be >= 1")

    n = uv.shape[0]
    rng = np.random.default_rng(random_seed)

    best_mask = None
    best_center = None
    best_radius = None
    best_score = -1
    best_mean_res = np.inf

    for _ in range(max_trials):
        idx = rng.choice(n, size=3, replace=False)
        center, radius = _circle_from_3_points(uv[idx[0]], uv[idx[1]], uv[idx[2]])
        if center is None:
            continue

        d = np.linalg.norm(uv - center[None, :], axis=1)
        residuals = np.abs(d - radius)
        mask = residuals <= residual_threshold
        score = int(mask.sum())
        if score == 0:
            continue
        mean_res = float(residuals[mask].mean())

        if (score > best_score) or (score == best_score and mean_res < best_mean_res):
            best_score = score
            best_mean_res = mean_res
            best_mask = mask
            best_center = center
            best_radius = radius

    if best_mask is None:
        raise RuntimeError("RANSAC failed to find a valid circle model")

    if min_inliers is not None and int(best_mask.sum()) < int(min_inliers):
        raise RuntimeError(
            f"RANSAC inliers below requirement: {int(best_mask.sum())} < {int(min_inliers)}"
        )

    if refit and best_mask.sum() >= 3:
        best_center, best_radius = _refit_circle_least_squares(uv[best_mask])

    residuals_all = np.abs(np.linalg.norm(uv - best_center[None, :], axis=1) - best_radius)
    final_mask = residuals_all <= residual_threshold
    inliers = uv[final_mask]

    return {
        "center": best_center,
        "radius": float(best_radius),
        "inlier_mask": final_mask,
        "inliers": inliers,
        "residuals": residuals_all,
        "num_inliers": int(final_mask.sum()),
        "inlier_ratio": float(final_mask.mean()),
    }

def find_symmetric_angle_from_circles(
    uv: np.ndarray,
    side_split_center: np.ndarray,
    left_center: np.ndarray,
    right_center: np.ndarray,
    left_radius: float,
    right_radius: float,
    j_left: np.ndarray,
    j_right: np.ndarray,
    tol: float,
    t_ratio: float = 0.5,
    min_bin_count: int = 30,
    step_deg: float = 1.0,
    max_angle_deg: float = 140.0,
    fail_tolerance: int = 1,
):
    """
    Common symmetric angle scan on left/right circles from given junctions.
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv, dtype=np.float64)
    c_split = np.asarray(side_split_center, dtype=np.float64).reshape(2)
    cl = np.asarray(left_center, dtype=np.float64).reshape(2)
    cr = np.asarray(right_center, dtype=np.float64).reshape(2)
    jl = np.asarray(j_left, dtype=np.float64).reshape(2)
    jr = np.asarray(j_right, dtype=np.float64).reshape(2)
    rl = float(left_radius)
    rr = float(right_radius)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if rl <= 0 or rr <= 0:
        raise ValueError("left_radius and right_radius must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if not (0.0 <= t_ratio <= 1.0):
        raise ValueError("t_ratio must satisfy 0 <= t_ratio <= 1")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be >= 1")
    if step_deg <= 0:
        raise ValueError("step_deg must be > 0")
    if max_angle_deg <= 0 or max_angle_deg > 179.0:
        raise ValueError("max_angle_deg must satisfy 0 < max_angle_deg <= 179")
    if fail_tolerance < 0:
        raise ValueError("fail_tolerance must be >= 0")

    # split by top-reference around side split center
    theta_top = np.pi / 2.0
    rel = uv - c_split[None, :]
    phi = wrap_pi(np.arctan2(rel[:, 1], rel[:, 0]) - theta_top)
    left_idx = np.where(phi > 0.0)[0]
    right_idx = np.where(phi < 0.0)[0]
    p_left = uv[left_idx]
    p_right = uv[right_idx]
    if p_left.shape[0] < 3 or p_right.shape[0] < 3:

        angle_rad = 0.0
        angle_deg = 0.0
        j_next_left = jl.copy()
        j_next_right = jr.copy()
        left_arc_mask = np.zeros(uv.shape[0], dtype=bool)
        right_arc_mask = np.zeros(uv.shape[0], dtype=bool)
        angle_span_mask = np.zeros(uv.shape[0], dtype=bool)
        remaining_mask = np.ones(uv.shape[0], dtype=bool)
        uv_remaining = uv.copy()

        return {
            "angle_deg": angle_deg,
            "angle_rad": angle_rad,
            "accepted_until_bin": -1,
            "bin_centers_deg": np.empty((0,), dtype=np.float64),
            "left_ratio": np.empty((0,), dtype=np.float64),
            "right_ratio": np.empty((0,), dtype=np.float64),
            "left_count": np.empty((0,), dtype=np.int32),
            "right_count": np.empty((0,), dtype=np.int32),
            "left_inlier_count": np.empty((0,), dtype=np.int32),
            "right_inlier_count": np.empty((0,), dtype=np.int32),
            "accepted_bins": np.empty((0,), dtype=bool),
            "left_arc_mask": left_arc_mask,
            "right_arc_mask": right_arc_mask,
            "angle_span_mask": angle_span_mask,
            "remaining_mask": remaining_mask,
            "uv_remaining": uv_remaining,
            "j_left": jl,
            "j_right": jr,
            "j_next_left": j_next_left,
            "j_next_right": j_next_right,
            "residuals_left": np.empty((0,), dtype=np.float64),
            "residuals_right": np.empty((0,), dtype=np.float64),
            "status": "fallback_not_enough_lr_points",
            "fallback_used": True,
        }

    # residual inlier masks
    res_l = np.abs(np.linalg.norm(p_left - cl[None, :], axis=1) - rl)
    res_r = np.abs(np.linalg.norm(p_right - cr[None, :], axis=1) - rr)
    in_l = res_l <= tol
    in_r = res_r <= tol

    # initial progression sign from junction tangent
    uj_l = (jl - cl) / rl
    uj_r = (jr - cr) / rr
    t1_l = np.array([-uj_l[1], uj_l[0]], dtype=np.float64)
    t2_l = -t1_l
    t1_r = np.array([-uj_r[1], uj_r[0]], dtype=np.float64)
    t2_r = -t1_r
    t_down_l = t1_l if t1_l[1] < t2_l[1] else t2_l
    t_down_r = t1_r if t1_r[1] < t2_r[1] else t2_r
    dir_sign_l = np.sign(cross2d(uj_l, t_down_l)) or 1.0
    dir_sign_r = np.sign(cross2d(uj_r, t_down_r)) or 1.0

    ang_l = np.arctan2((p_left - cl[None, :])[:, 1], (p_left - cl[None, :])[:, 0])
    ang_r = np.arctan2((p_right - cr[None, :])[:, 1], (p_right - cr[None, :])[:, 0])
    ang_j_l = np.arctan2((jl - cl)[1], (jl - cl)[0])
    ang_j_r = np.arctan2((jr - cr)[1], (jr - cr)[0])

    raw_l = wrap_pi(ang_l - ang_j_l)
    raw_r = wrap_pi(ang_r - ang_j_r)

    # Choose sign per side so that the "forward" direction matches the dominant side progression.
    # This fixes A1 top-junction case where tangent-based sign can flip one side.
    def choose_sign(raw: np.ndarray, fallback: float) -> float:
        pos = int(np.count_nonzero(raw >= 0.0))
        neg = int(np.count_nonzero(raw <= 0.0))
        if pos > neg:
            return 1.0
        if neg > pos:
            return -1.0
        return float(fallback)

    dir_sign_l = choose_sign(raw_l, dir_sign_l)
    dir_sign_r = choose_sign(raw_r, dir_sign_r)
    beta_l = dir_sign_l * raw_l
    beta_r = dir_sign_r * raw_r
    valid_l = beta_l >= 0.0
    valid_r = beta_r >= 0.0

    step = np.deg2rad(step_deg)
    amax = np.deg2rad(max_angle_deg)
    edges = np.arange(0.0, amax + step, step, dtype=np.float64)
    if edges.shape[0] < 2:
        raise ValueError("No bins generated; check step_deg/max_angle_deg")
    centers = 0.5 * (edges[:-1] + edges[1:])
    k = centers.shape[0]

    left_ratio = np.zeros(k, dtype=np.float64)
    right_ratio = np.zeros(k, dtype=np.float64)
    left_count = np.zeros(k, dtype=np.int32)
    right_count = np.zeros(k, dtype=np.int32)
    left_inlier_count = np.zeros(k, dtype=np.int32)
    right_inlier_count = np.zeros(k, dtype=np.int32)
    accepted_bins = np.zeros(k, dtype=bool)

    accepted_until = -1
    consecutive_fail = 0
    has_started = False
    
    for i in range(k):
        b0 = edges[i]
        b1 = edges[i + 1]
        lb = valid_l & (beta_l >= b0) & (beta_l < b1)
        rb = valid_r & (beta_r >= b0) & (beta_r < b1)
        lc = int(lb.sum())
        rc = int(rb.sum())
        left_count[i] = lc
        right_count[i] = rc
        li = int(in_l[lb].sum()) if lc > 0 else 0
        ri = int(in_r[rb].sum()) if rc > 0 else 0
        left_inlier_count[i] = li
        right_inlier_count[i] = ri
        lr = float(in_l[lb].mean()) if lc > 0 else 0.0
        rr_ = float(in_r[rb].mean()) if rc > 0 else 0.0
        left_ratio[i] = lr
        right_ratio[i] = rr_
        ok = (lc >= min_bin_count) and (rc >= min_bin_count) and (lr >= t_ratio) and (rr_ >= t_ratio)
        accepted_bins[i] = ok
        if ok:
            accepted_until = i
            consecutive_fail = 0
            has_started = True
        else:
            if has_started:
                consecutive_fail += 1
                if consecutive_fail > fail_tolerance:
                    break

    angle_rad = float(edges[accepted_until + 1]) if accepted_until >= 0 else 0.0
    angle_deg = float(np.rad2deg(angle_rad))

    # endpoint junctions at selected symmetric angle
    ang_next_l = ang_j_l + dir_sign_l * angle_rad
    ang_next_r = ang_j_r + dir_sign_r * angle_rad
    j_next_left = cl + rl * np.array([np.cos(ang_next_l), np.sin(ang_next_l)], dtype=np.float64)
    j_next_right = cr + rr * np.array([np.cos(ang_next_r), np.sin(ang_next_r)], dtype=np.float64)

    left_arc_mask = np.zeros(uv.shape[0], dtype=bool)
    right_arc_mask = np.zeros(uv.shape[0], dtype=bool)
    left_span_mask = np.zeros(uv.shape[0], dtype=bool)
    right_span_mask = np.zeros(uv.shape[0], dtype=bool)
    left_arc_mask[left_idx] = valid_l & (beta_l <= angle_rad) & in_l
    right_arc_mask[right_idx] = valid_r & (beta_r <= angle_rad) & in_r
    left_span_mask[left_idx] = valid_l & (beta_l <= angle_rad)
    right_span_mask[right_idx] = valid_r & (beta_r <= angle_rad)
    angle_span_mask = left_span_mask | right_span_mask
    remaining_mask = ~angle_span_mask
    uv_remaining = uv[remaining_mask]

    return {
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
        "accepted_until_bin": int(accepted_until),
        "bin_centers_deg": np.rad2deg(centers),
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "left_count": left_count,
        "right_count": right_count,
        "left_inlier_count": left_inlier_count,
        "right_inlier_count": right_inlier_count,
        "accepted_bins": accepted_bins,
        "left_arc_mask": left_arc_mask,
        "right_arc_mask": right_arc_mask,
        "angle_span_mask": angle_span_mask,
        "remaining_mask": remaining_mask,
        "uv_remaining": uv_remaining,
        "j_left": jl,
        "j_right": jr,
        "j_next_left": j_next_left,
        "j_next_right": j_next_right,
        "residuals_left": res_l,
        "residuals_right": res_r,
    }

def find_symmetric_a1_from_o1(
    uv: np.ndarray,
    center: np.ndarray,
    radius: float,
    tol1: float,
    t_ratio: float = 0.5,
    min_bin_count: int = 30,
    step_deg: float = 1.0,
    max_alpha_deg: float = 120.0,
    fail_tolerance: int = 0,
):
    """
    Find symmetric A1 by scanning from top toward left/right simultaneously.

    Angle definition:
    - theta = atan2(v-cy, u-cx)
    - top direction is +v (theta_top = +pi/2)
    - relative angle phi = wrap(theta - theta_top) in [-pi, pi]
      left side: phi > 0, right side: phi < 0
    - symmetric O1 range is [-A1, +A1] around top.

    A bin at offset alpha is accepted when BOTH sides satisfy:
    - bin point count >= min_bin_count
    - inlier ratio (|dist-center - radius| <= tol1) >= t_ratio

    Returns:
        {
            "A1_deg": float,
            "A1_rad": float,
            "accepted_until_bin": int,
            "alpha_centers_deg": (K,),
            "left_ratio": (K,),
            "right_ratio": (K,),
            "left_count": (K,),
            "right_count": (K,),
            "accepted_bins": (K,) bool,
            "o1_mask_symmetric": (N,) bool,
            "a1_span_mask": (N,) bool,
            "remaining_mask": (N,) bool,
            "uv_without_a1": (M,2),
            "residuals": (N,),
        }
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(2)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if tol1 <= 0:
        raise ValueError("tol1 must be > 0")
    if not (0.0 <= t_ratio <= 1.0):
        raise ValueError("t_ratio must satisfy 0 <= t_ratio <= 1")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be >= 1")
    if step_deg <= 0:
        raise ValueError("step_deg must be > 0")
    if max_alpha_deg <= 0 or max_alpha_deg > 179.0:
        raise ValueError("max_alpha_deg must satisfy 0 < max_alpha_deg <= 179")
    if fail_tolerance < 0:
        raise ValueError("fail_tolerance must be >= 0")

    theta_top = np.pi / 2.0
    j_top = center + float(radius) * np.array([np.cos(theta_top), np.sin(theta_top)], dtype=np.float64)
    common = find_symmetric_angle_from_circles(
        uv=uv,
        side_split_center=center,
        left_center=center,
        right_center=center,
        left_radius=float(radius),
        right_radius=float(radius),
        j_left=j_top,
        j_right=j_top,
        tol=float(tol1),
        t_ratio=t_ratio,
        min_bin_count=min_bin_count,
        step_deg=step_deg,
        max_angle_deg=max_alpha_deg,
        fail_tolerance=fail_tolerance,
    )
    # full-circle residual for compatibility
    rel = uv - center[None, :]
    residuals = np.abs(np.linalg.norm(rel, axis=1) - float(radius))
    return {
        "A1_deg": common["angle_deg"],
        "A1_rad": common["angle_rad"],
        "j1_left": common["j_next_left"],
        "j1_right": common["j_next_right"],
        "accepted_until_bin": common["accepted_until_bin"],
        "alpha_centers_deg": common["bin_centers_deg"],
        "left_ratio": common["left_ratio"],
        "right_ratio": common["right_ratio"],
        "left_count": common["left_count"],
        "right_count": common["right_count"],
        "left_inlier_count": common["left_inlier_count"],
        "right_inlier_count": common["right_inlier_count"],
        "accepted_bins": common["accepted_bins"],
        "o1_mask_symmetric": common["left_arc_mask"] | common["right_arc_mask"],
        "a1_span_mask": common["angle_span_mask"],
        "remaining_mask": common["remaining_mask"],
        "uv_without_a1": common["uv_remaining"],
        "residuals": residuals,
    }

    rel = uv - center[None, :]
    d = np.linalg.norm(rel, axis=1)
    residuals = np.abs(d - float(radius))
    inlier = residuals <= float(tol1)

    theta = np.arctan2(rel[:, 1], rel[:, 0])
    theta_top = np.pi / 2.0
    phi = np.arctan2(np.sin(theta - theta_top), np.cos(theta - theta_top))

    step = np.deg2rad(step_deg)
    alpha_max = np.deg2rad(max_alpha_deg)
    edges = np.arange(0.0, alpha_max + step, step, dtype=np.float64)
    if edges.shape[0] < 2:
        raise ValueError("No bins generated; check step_deg/max_alpha_deg")

    alpha_centers = 0.5 * (edges[:-1] + edges[1:])
    k = alpha_centers.shape[0]

    left_ratio = np.zeros(k, dtype=np.float64)
    right_ratio = np.zeros(k, dtype=np.float64)
    left_count = np.zeros(k, dtype=np.int32)
    right_count = np.zeros(k, dtype=np.int32)
    accepted_bins = np.zeros(k, dtype=bool)

    consecutive_fail = 0
    accepted_until = -1

    for i in range(k):
        a0 = edges[i]
        a1 = edges[i + 1]

        left_mask = (phi >= a0) & (phi < a1)
        right_mask = (phi <= -a0) & (phi > -a1)

        lc = int(left_mask.sum())
        rc = int(right_mask.sum())
        left_count[i] = lc
        right_count[i] = rc

        lr = float(inlier[left_mask].mean()) if lc > 0 else 0.0
        rr = float(inlier[right_mask].mean()) if rc > 0 else 0.0
        left_ratio[i] = lr
        right_ratio[i] = rr

        ok_left = (lc >= min_bin_count) and (lr >= t_ratio)
        ok_right = (rc >= min_bin_count) and (rr >= t_ratio)
        ok = ok_left and ok_right
        accepted_bins[i] = ok

        if ok:
            accepted_until = i
            consecutive_fail = 0
        else:
            consecutive_fail += 1
            if consecutive_fail > fail_tolerance:
                break

    if accepted_until >= 0:
        a1_rad = float(edges[accepted_until + 1])
    else:
        a1_rad = 0.0
    a1_deg = float(np.rad2deg(a1_rad))

    # J1 junctions at +/-A1 on O1
    th_l = theta_top + a1_rad
    th_r = theta_top - a1_rad
    j1_left = center + float(radius) * np.array([np.cos(th_l), np.sin(th_l)], dtype=np.float64)
    j1_right = center + float(radius) * np.array([np.cos(th_r), np.sin(th_r)], dtype=np.float64)

    # O1 inlier mask inside symmetric A1 span
    o1_mask_symmetric = (np.abs(phi) <= a1_rad) & inlier
    # Full A1 angular span mask (regardless of O1 radial inlier)
    a1_span_mask = np.abs(phi) <= a1_rad
    remaining_mask = ~a1_span_mask
    uv_without_a1 = uv[remaining_mask]

    return {
        "A1_deg": a1_deg,
        "A1_rad": a1_rad,
        "j1_left": j1_left,
        "j1_right": j1_right,
        "accepted_until_bin": int(accepted_until),
        "alpha_centers_deg": np.rad2deg(alpha_centers),
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "left_count": left_count,
        "right_count": right_count,
        "accepted_bins": accepted_bins,
        "o1_mask_symmetric": o1_mask_symmetric,
        "a1_span_mask": a1_span_mask,
        "remaining_mask": remaining_mask,
        "uv_without_a1": uv_without_a1,
        "residuals": residuals,
    }

def fit_o2_symmetric_constrained_ransac(
    uv_without_a1: np.ndarray,
    o1_center: np.ndarray,
    o1_radius: float,
    a1_deg: float,
    residual_threshold: float,
    max_trials: int = 4000,
    min_inliers_per_side: int = 50,
    random_seed: int | None = 0,
    refit: bool = True,
    enforce_c2_between_c1_and_j: bool = True,
    r_min: float | None = None,
    r_hi: float | None = None,
    fallback_eps: float = 0.05,
):
    """
    Fit symmetric O2L/O2R circles under geometric constraints using 1-parameter RANSAC.

    Constraints:
    - O2 centers are on lines (O1_center -> O1/O2 junction points at +/-A1)
      and placed toward O1 center side from the junction
    - O2L and O2R share the same radius r2
    - c2L = JL - nL * r2, c2R = JR - nR * r2
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv_without_a1, dtype=np.float64)
    c1 = np.asarray(o1_center, dtype=np.float64).reshape(2)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv_without_a1 must have shape (N, 2)")
    if o1_radius <= 0:
        raise ValueError("o1_radius must be > 0")

    a1_rad = np.deg2rad(float(a1_deg))
    theta_top = np.pi / 2.0
    th_l = theta_top + a1_rad
    th_r = theta_top - a1_rad
    n_l = np.array([np.cos(th_l), np.sin(th_l)], dtype=np.float64)
    n_r = np.array([np.cos(th_r), np.sin(th_r)], dtype=np.float64)
    j_l = c1 + float(o1_radius) * n_l
    j_r = c1 + float(o1_radius) * n_r

    r_min_eff = 0.0 if r_min is None else float(r_min)
    if r_min_eff < 0:
        raise ValueError("r_min must be >= 0")
    if r_hi is not None:
        r_hi_eff = float(r_hi)
    else:
        r_hi_eff = float(o1_radius) if enforce_c2_between_c1_and_j else None
    if r_hi_eff is not None and not (r_hi_eff > r_min_eff):
        raise ValueError(
            f"Invalid O2 search range: r_min={r_min_eff}, r_max={r_hi_eff}. "
            "Check r_min/r_hi and o1_radius."
        )

    # Pre-check left/right candidate counts with same split rule used by common fit.
    rel = uv - c1[None, :]
    phi = wrap_pi(np.arctan2(rel[:, 1], rel[:, 0]) - theta_top)
    left_candidate_mask = phi > 0.0
    right_candidate_mask = phi < 0.0

    if int(left_candidate_mask.sum()) < 3 or int(right_candidate_mask.sum()) < 3:
        # Fallback for near-circular/fully-covered case where no O2/O3 band remains.
        r2_fb = float(o1_radius)
        c2_l_fb = j_l - n_l * r2_fb
        c2_r_fb = j_r - n_r * r2_fb
        empty_mask = np.zeros(uv.shape[0], dtype=bool)
        empty_pts = np.empty((0, 2), dtype=np.float64)
        return {
            "r2": float(r2_fb),
            "o2_left_center": c2_l_fb,
            "o2_right_center": c2_r_fb,
            "o1_mask_removed": np.zeros(uv.shape[0], dtype=bool),
            "candidate_mask": np.ones(uv.shape[0], dtype=bool),
            "left_candidate_mask": left_candidate_mask,
            "right_candidate_mask": right_candidate_mask,
            "o2_left_inlier_mask": empty_mask,
            "o2_right_inlier_mask": empty_mask,
            "o2_left_inliers": empty_pts,
            "o2_right_inliers": empty_pts,
            "left_residuals_on_candidates": np.empty((0,), dtype=np.float64),
            "right_residuals_on_candidates": np.empty((0,), dtype=np.float64),
            "left_inlier_count": 0,
            "right_inlier_count": 0,
            "status": "fallback_not_enough_lr_points",
            "fallback_used": True,
        }

    try:
        res = fit_symmetric_constrained_circle_ransac(
            uv=uv,
            side_split_center=c1,
            j_left=j_l,
            j_right=j_r,
            n_left=n_l,
            n_right=n_r,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            min_inliers_per_side=min_inliers_per_side,
            random_seed=random_seed,
            refit=refit,
            r_min=r_min_eff,
            r_max=r_hi_eff,
            sign_mode="minus",
        )
    except RuntimeError as e:
        # Fallback instead of hard stop for sparse/degenerate candidates.
        r2_fb = float(o1_radius)
        c2_l_fb = j_l - n_l * r2_fb
        c2_r_fb = j_r - n_r * r2_fb
        empty_mask = np.zeros(uv.shape[0], dtype=bool)
        empty_pts = np.empty((0, 2), dtype=np.float64)
        return {
            "r2": float(r2_fb),
            "o2_left_center": c2_l_fb,
            "o2_right_center": c2_r_fb,
            "o1_mask_removed": np.zeros(uv.shape[0], dtype=bool),
            "candidate_mask": np.ones(uv.shape[0], dtype=bool),
            "left_candidate_mask": left_candidate_mask,
            "right_candidate_mask": right_candidate_mask,
            "o2_left_inlier_mask": empty_mask,
            "o2_right_inlier_mask": empty_mask,
            "o2_left_inliers": empty_pts,
            "o2_right_inliers": empty_pts,
            "left_residuals_on_candidates": np.empty((0,), dtype=np.float64),
            "right_residuals_on_candidates": np.empty((0,), dtype=np.float64),
            "left_inlier_count": 0,
            "right_inlier_count": 0,
            "status": f"fallback_runtime_error: {str(e)}",
            "fallback_used": True,
        }

    return {
        "r2": float(res["r"]),
        "o2_left_center": res["left_center"],
        "o2_right_center": res["right_center"],
        "o1_mask_removed": np.zeros(uv.shape[0], dtype=bool),
        "candidate_mask": np.ones(uv.shape[0], dtype=bool),
        "left_candidate_mask": res["left_candidate_mask"],
        "right_candidate_mask": res["right_candidate_mask"],
        "o2_left_inlier_mask": res["left_inlier_mask"],
        "o2_right_inlier_mask": res["right_inlier_mask"],
        "o2_left_inliers": res["left_inliers"],
        "o2_right_inliers": res["right_inliers"],
        "left_residuals_on_candidates": res["left_residuals_on_candidates"],
        "right_residuals_on_candidates": res["right_residuals_on_candidates"],
        "left_inlier_count": int(res["left_inlier_count"]),
        "right_inlier_count": int(res["right_inlier_count"]),
        "status": "ok",
        "fallback_used": False,
    }

def fit_symmetric_constrained_circle_ransac(
    uv: np.ndarray,
    side_split_center: np.ndarray,
    j_left: np.ndarray,
    j_right: np.ndarray,
    n_left: np.ndarray,
    n_right: np.ndarray,
    residual_threshold: float,
    max_trials: int = 4000,
    min_inliers_per_side: int = 10,
    random_seed: int | None = 0,
    refit: bool = True,
    r_min: float = 0.0,
    r_max: float | None = None,
    sign_mode: str = "auto",
):
    """
    Generic symmetric constrained-circle RANSAC (shared radius for left/right).

    Model:
    - c_left  = j_left  + s * n_left  * r
    - c_right = j_right + s * n_right * r
    where s in {-1, +1} (minus/plus direction), and r is shared.
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv, dtype=np.float64)
    c_split = np.asarray(side_split_center, dtype=np.float64).reshape(2)
    jl = np.asarray(j_left, dtype=np.float64).reshape(2)
    jr = np.asarray(j_right, dtype=np.float64).reshape(2)
    nl = np.asarray(n_left, dtype=np.float64).reshape(2)
    nr = np.asarray(n_right, dtype=np.float64).reshape(2)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if residual_threshold <= 0:
        raise ValueError("residual_threshold must be > 0")
    if max_trials < 1:
        raise ValueError("max_trials must be >= 1")
    if min_inliers_per_side < 1:
        raise ValueError("min_inliers_per_side must be >= 1")
    if r_min < 0:
        raise ValueError("r_min must be >= 0")
    if r_max is not None and r_max <= r_min:
        raise ValueError("r_max must be > r_min when provided")
    if sign_mode not in ("auto", "minus", "plus"):
        raise ValueError("sign_mode must be one of {'auto', 'minus', 'plus'}")

    nl_norm = np.linalg.norm(nl)
    nr_norm = np.linalg.norm(nr)
    if nl_norm <= 1e-12 or nr_norm <= 1e-12:
        raise ValueError("n_left and n_right must be non-zero vectors")
    nl = nl / nl_norm
    nr = nr / nr_norm

    # left/right split by +v top reference around side_split_center
    theta_top = np.pi / 2.0
    rel = uv - c_split[None, :]
    phi = wrap_pi(np.arctan2(rel[:, 1], rel[:, 0]) - theta_top)
    left_mask = phi > 0.0
    right_mask = phi < 0.0
    p_left = uv[left_mask]
    p_right = uv[right_mask]
    if p_left.shape[0] < 3 or p_right.shape[0] < 3:
        raise RuntimeError("Not enough left/right candidate points")

    def evaluate_r(r: float, sign: float):
        c_l = jl + sign * nl * r
        c_r = jr + sign * nr * r
        res_l = np.abs(np.linalg.norm(p_left - c_l[None, :], axis=1) - r)
        res_r = np.abs(np.linalg.norm(p_right - c_r[None, :], axis=1) - r)
        in_l = res_l <= residual_threshold
        in_r = res_r <= residual_threshold
        cnt_l = int(in_l.sum())
        cnt_r = int(in_r.sum())
        cnt_min = min(cnt_l, cnt_r)
        cnt_sum = cnt_l + cnt_r
        mean_in = np.inf
        if cnt_sum > 0:
            vals = []
            if cnt_l > 0:
                vals.append(float(res_l[in_l].mean()))
            if cnt_r > 0:
                vals.append(float(res_r[in_r].mean()))
            mean_in = float(np.mean(vals))
        return {
            "r": float(r),
            "sign": float(sign),
            "c_left": c_l,
            "c_right": c_r,
            "res_l": res_l,
            "res_r": res_r,
            "in_l": in_l,
            "in_r": in_r,
            "cnt_l": cnt_l,
            "cnt_r": cnt_r,
            "cnt_min": cnt_min,
            "cnt_sum": cnt_sum,
            "mean_in": mean_in,
        }

    def better(cur: dict, best: dict) -> bool:
        return (
            cur["cnt_min"] > best["cnt_min"]
            or (
                cur["cnt_min"] == best["cnt_min"]
                and (
                    cur["cnt_sum"] > best["cnt_sum"]
                    or (
                        cur["cnt_sum"] == best["cnt_sum"]
                        and cur["mean_in"] < best["mean_in"]
                    )
                )
            )
        )

    def run_for_sign(sign: float):
        # Pure range scan over r_min~r_max.
        r_lo = float(r_min)
        if r_max is not None:
            r_hi = float(r_max)
        else:
            # Fallback upper bound if user did not provide r_max.
            r_hi = float(
                max(
                    np.linalg.norm(p_left - jl[None, :], axis=1).max(initial=0.0),
                    np.linalg.norm(p_right - jr[None, :], axis=1).max(initial=0.0),
                )
            )
        if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
            return None

        # max_trials is reused as sweep resolution.
        n_grid = int(np.clip(max_trials, 100, 5000))
        r_grid = np.linspace(r_lo, r_hi, n_grid, dtype=np.float64)

        best_local = None
        for r in r_grid:
            cur = evaluate_r(float(r), sign)
            if best_local is None or better(cur, best_local):
                best_local = cur
        if best_local is None:
            return None

        if refit:
            # Local refinement around best r
            dr = (r_hi - r_lo) / max(n_grid - 1, 1)
            rr_lo = max(float(r_min), float(best_local["r"] - 5.0 * dr))
            rr_hi = min(float(r_max), float(best_local["r"] + 5.0 * dr)) if r_max is not None else float(best_local["r"] + 5.0 * dr)
            if rr_hi > rr_lo:
                r_ref = np.linspace(rr_lo, rr_hi, 200, dtype=np.float64)
                for r in r_ref:
                    cur = evaluate_r(float(r), sign)
                    if better(cur, best_local):
                        best_local = cur

        return best_local

    signs = [-1.0, +1.0] if sign_mode == "auto" else ([-1.0] if sign_mode == "minus" else [+1.0])
    # 6) Apply reject rules / penalties and select the best model.
    candidates = []
    for s in signs:
        m = run_for_sign(s)
        if m is not None:
            candidates.append(m)

    if not candidates:
        raise RuntimeError("Constrained RANSAC failed for all allowed sign directions")

    best = candidates[0]
    for cur in candidates[1:]:
        if better(cur, best):
            best = cur

    if best["cnt_l"] < min_inliers_per_side or best["cnt_r"] < min_inliers_per_side:
        raise RuntimeError(
            f"Inliers too small: left={best['cnt_l']}, right={best['cnt_r']}, required={min_inliers_per_side}"
        )

    inlier_left_full = np.zeros(uv.shape[0], dtype=bool)
    inlier_right_full = np.zeros(uv.shape[0], dtype=bool)
    inlier_left_full[np.where(left_mask)[0]] = best["in_l"]
    inlier_right_full[np.where(right_mask)[0]] = best["in_r"]

    return {
        "r": float(best["r"]),
        "sign": float(best["sign"]),
        "left_center": best["c_left"],
        "right_center": best["c_right"],
        "left_candidate_mask": left_mask,
        "right_candidate_mask": right_mask,
        "left_inlier_mask": inlier_left_full,
        "right_inlier_mask": inlier_right_full,
        "left_inliers": uv[inlier_left_full],
        "right_inliers": uv[inlier_right_full],
        "left_residuals_on_candidates": best["res_l"],
        "right_residuals_on_candidates": best["res_r"],
        "left_inlier_count": int(best["cnt_l"]),
        "right_inlier_count": int(best["cnt_r"]),
    }

def fit_o3_symmetric_constrained_ransac(
    uv_without_a2: np.ndarray,
    o1_center: np.ndarray,
    o1_radius: float,
    a1_deg: float,
    o2_left_center: np.ndarray,
    o2_right_center: np.ndarray,
    o2_radius: float,
    a2_deg: float,
    residual_threshold: float,
    max_trials: int = 4000,
    min_inliers_per_side: int = 10,
    random_seed: int | None = 0,
    refit: bool = True,
    enforce_r3_gt_r2: bool = True,
    r_min: float | None = None,
    r_hi: float | None = None,
    fallback_eps: float = 0.05,
):
    """
    Fit symmetric O3L/O3R circles from uv_without_a2 with constrained RANSAC.

    Constraints:
    - C2, J2, C3 are collinear with fixed center direction:
      C3 = J2 - n*r3  (C3-C2-J2 order).
    - Left/right share common radius r3.
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv_without_a2, dtype=np.float64)
    c1 = np.asarray(o1_center, dtype=np.float64).reshape(2)
    c2l = np.asarray(o2_left_center, dtype=np.float64).reshape(2)
    c2r = np.asarray(o2_right_center, dtype=np.float64).reshape(2)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv_without_a2 must have shape (N, 2)")
    if o1_radius <= 0 or o2_radius <= 0:
        raise ValueError("o1_radius and o2_radius must be > 0")

    theta_top = np.pi / 2.0
    a1_rad = np.deg2rad(float(a1_deg))
    a2_rad = np.deg2rad(float(a2_deg))

    # J1 points from O1
    th_l = theta_top + a1_rad
    th_r = theta_top - a1_rad
    n1_l = np.array([np.cos(th_l), np.sin(th_l)], dtype=np.float64)
    n1_r = np.array([np.cos(th_r), np.sin(th_r)], dtype=np.float64)
    j1_l = c1 + float(o1_radius) * n1_l
    j1_r = c1 + float(o1_radius) * n1_r

    # J2 points from O2 and A2
    ujl = (j1_l - c2l) / float(o2_radius)
    ujr = (j1_r - c2r) / float(o2_radius)
    t1_l = np.array([-ujl[1], ujl[0]], dtype=np.float64)
    t2_l = -t1_l
    t1_r = np.array([-ujr[1], ujr[0]], dtype=np.float64)
    t2_r = -t1_r
    t_down_l = t1_l if t1_l[1] < t2_l[1] else t2_l
    t_down_r = t1_r if t1_r[1] < t2_r[1] else t2_r
    dir_sign_l = np.sign(cross2d(ujl, t_down_l)) or 1.0
    dir_sign_r = np.sign(cross2d(ujr, t_down_r)) or 1.0
    ang_j1_l = np.arctan2((j1_l - c2l)[1], (j1_l - c2l)[0])
    ang_j1_r = np.arctan2((j1_r - c2r)[1], (j1_r - c2r)[0])
    ang_j2_l = ang_j1_l + dir_sign_l * a2_rad
    ang_j2_r = ang_j1_r + dir_sign_r * a2_rad
    j2_l = c2l + float(o2_radius) * np.array([np.cos(ang_j2_l), np.sin(ang_j2_l)], dtype=np.float64)
    j2_r = c2r + float(o2_radius) * np.array([np.cos(ang_j2_r), np.sin(ang_j2_r)], dtype=np.float64)
    n2_l = (j2_l - c2l) / float(o2_radius)
    n2_r = (j2_r - c2r) / float(o2_radius)

    if r_min is not None:
        r_min_eff = float(r_min)
    else:
        r_min_eff = 0.0
    if r_min_eff < 0:
        raise ValueError("r_min must be >= 0")
    if enforce_r3_gt_r2:
        # If the strict relation is requested and caller did not set r_min,
        # default lower bound starts at O2 radius.
        r_min_eff = max(float(r_min_eff), float(o2_radius))

    if r_hi is None and enforce_r3_gt_r2:
        raise ValueError("r_hi must be provided when enforce_r3_gt_r2=True")
    r_hi_eff = float(r_hi) if r_hi is not None else None
    if r_hi_eff is not None:
        if not (r_hi_eff > r_min_eff):
            raise ValueError(
                f"Invalid O3 search range: r_min={r_min_eff}, r_max={r_hi_eff}. "
                "Check r_min/r_hi and o2_radius."
            )

    # Pre-check left/right candidate counts with same split rule used by common fit.
    rel = uv - c1[None, :]
    phi = wrap_pi(np.arctan2(rel[:, 1], rel[:, 0]) - theta_top)
    left_candidate_mask = phi > 0.0
    right_candidate_mask = phi < 0.0

    if int(left_candidate_mask.sum()) < 3 or int(right_candidate_mask.sum()) < 3:
        r3_fb = float(o2_radius)
        c3_l_fb = j2_l - n2_l * r3_fb
        c3_r_fb = j2_r - n2_r * r3_fb
        empty_mask = np.zeros(uv.shape[0], dtype=bool)
        empty_pts = np.empty((0, 2), dtype=np.float64)
        return {
            "r3": float(r3_fb),
            "o3_left_center": c3_l_fb,
            "o3_right_center": c3_r_fb,
            "j2_left": j2_l,
            "j2_right": j2_r,
            "left_candidate_mask": left_candidate_mask,
            "right_candidate_mask": right_candidate_mask,
            "o3_left_inlier_mask": empty_mask,
            "o3_right_inlier_mask": empty_mask,
            "o3_left_inliers": empty_pts,
            "o3_right_inliers": empty_pts,
            "left_residuals_on_candidates": np.empty((0,), dtype=np.float64),
            "right_residuals_on_candidates": np.empty((0,), dtype=np.float64),
            "left_inlier_count": 0,
            "right_inlier_count": 0,
            "status": "fallback_not_enough_lr_points",
            "fallback_used": True,
        }

    try:
        res = fit_symmetric_constrained_circle_ransac(
            uv=uv,
            side_split_center=c1,
            j_left=j2_l,
            j_right=j2_r,
            n_left=n2_l,
            n_right=n2_r,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            min_inliers_per_side=min_inliers_per_side,
            random_seed=random_seed,
            refit=refit,
            r_min=r_min_eff,
            r_max=r_hi_eff,
            sign_mode="minus",
        )
    except RuntimeError as e:
        r3_fb = float(o2_radius)
        c3_l_fb = j2_l - n2_l * r3_fb
        c3_r_fb = j2_r - n2_r * r3_fb
        empty_mask = np.zeros(uv.shape[0], dtype=bool)
        empty_pts = np.empty((0, 2), dtype=np.float64)
        return {
            "r3": float(r3_fb),
            "o3_left_center": c3_l_fb,
            "o3_right_center": c3_r_fb,
            "j2_left": j2_l,
            "j2_right": j2_r,
            "left_candidate_mask": left_candidate_mask,
            "right_candidate_mask": right_candidate_mask,
            "o3_left_inlier_mask": empty_mask,
            "o3_right_inlier_mask": empty_mask,
            "o3_left_inliers": empty_pts,
            "o3_right_inliers": empty_pts,
            "left_residuals_on_candidates": np.empty((0,), dtype=np.float64),
            "right_residuals_on_candidates": np.empty((0,), dtype=np.float64),
            "left_inlier_count": 0,
            "right_inlier_count": 0,
            "status": f"fallback_runtime_error: {str(e)}",
            "fallback_used": True,
        }

    return {
        "r3": float(res["r"]),
        "o3_left_center": res["left_center"],
        "o3_right_center": res["right_center"],
        "j2_left": j2_l,
        "j2_right": j2_r,
        "left_candidate_mask": res["left_candidate_mask"],
        "right_candidate_mask": res["right_candidate_mask"],
        "o3_left_inlier_mask": res["left_inlier_mask"],
        "o3_right_inlier_mask": res["right_inlier_mask"],
        "o3_left_inliers": res["left_inliers"],
        "o3_right_inliers": res["right_inliers"],
        "left_residuals_on_candidates": res["left_residuals_on_candidates"],
        "right_residuals_on_candidates": res["right_residuals_on_candidates"],
        "left_inlier_count": int(res["left_inlier_count"]),
        "right_inlier_count": int(res["right_inlier_count"]),
        "status": "ok",
        "fallback_used": False,
    }

def find_symmetric_a2_from_o2(
    uv_without_a1: np.ndarray,
    o1_center: np.ndarray,
    o1_radius: float,
    a1_deg: float,
    o2_left_center: np.ndarray,
    o2_right_center: np.ndarray,
    o2_radius: float,
    tol2: float,
    t_ratio: float = 0.5,
    min_bin_count: int = 30,
    step_deg: float = 1.0,
    max_a2_deg: float = 140.0,
    fail_tolerance: int = 1,
):
    """
    Find symmetric A2 from O2 circles by scanning down from O1/O2 junctions.

    - Input points are expected to be already outside A1 span (uv_without_a1).
    - Left/right are evaluated together per angular bin from each junction.
    - A2 is the largest symmetric accepted angle.

    Returns:
        {
            "A2_deg", "A2_rad",
            "accepted_until_bin",
            "beta_centers_deg",
            "left_ratio", "right_ratio",
            "left_count", "right_count",
            "accepted_bins",
            "o2_arc_mask",
            "o2_left_arc_mask",
            "o2_right_arc_mask",
            "a2_span_mask",
            "remaining_mask",
            "uv_without_a2",
        }
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv_without_a1, dtype=np.float64)
    c1 = np.asarray(o1_center, dtype=np.float64).reshape(2)
    c2l = np.asarray(o2_left_center, dtype=np.float64).reshape(2)
    c2r = np.asarray(o2_right_center, dtype=np.float64).reshape(2)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if o1_radius <= 0 or o2_radius <= 0:
        raise ValueError("o1_radius and o2_radius must be > 0")
    if tol2 <= 0:
        raise ValueError("tol2 must be > 0")
    if not (0.0 <= t_ratio <= 1.0):
        raise ValueError("t_ratio must satisfy 0 <= t_ratio <= 1")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be >= 1")
    if step_deg <= 0:
        raise ValueError("step_deg must be > 0")
    if max_a2_deg <= 0 or max_a2_deg > 179.0:
        raise ValueError("max_a2_deg must satisfy 0 < max_a2_deg <= 179")
    if fail_tolerance < 0:
        raise ValueError("fail_tolerance must be >= 0")

    theta_top = np.pi / 2.0
    a1_rad = np.deg2rad(float(a1_deg))
    th_l = theta_top + a1_rad
    th_r = theta_top - a1_rad
    n_l = np.array([np.cos(th_l), np.sin(th_l)], dtype=np.float64)
    n_r = np.array([np.cos(th_r), np.sin(th_r)], dtype=np.float64)
    j_l = c1 + float(o1_radius) * n_l
    j_r = c1 + float(o1_radius) * n_r

    common = find_symmetric_angle_from_circles(
        uv=uv,
        side_split_center=c1,
        left_center=c2l,
        right_center=c2r,
        left_radius=float(o2_radius),
        right_radius=float(o2_radius),
        j_left=j_l,
        j_right=j_r,
        tol=float(tol2),
        t_ratio=t_ratio,
        min_bin_count=min_bin_count,
        step_deg=step_deg,
        max_angle_deg=max_a2_deg,
        fail_tolerance=fail_tolerance,
    )
    return {
        "A2_deg": common["angle_deg"],
        "A2_rad": common["angle_rad"],
        "j1_left": common["j_left"],
        "j1_right": common["j_right"],
        "j2_left": common["j_next_left"],
        "j2_right": common["j_next_right"],
        "accepted_until_bin": common["accepted_until_bin"],
        "beta_centers_deg": common["bin_centers_deg"],
        "left_ratio": common["left_ratio"],
        "right_ratio": common["right_ratio"],
        "left_count": common["left_count"],
        "right_count": common["right_count"],
        "left_inlier_count": common["left_inlier_count"],
        "right_inlier_count": common["right_inlier_count"],
        "accepted_bins": common["accepted_bins"],
        "o2_arc_mask": common["left_arc_mask"] | common["right_arc_mask"],
        "o2_left_arc_mask": common["left_arc_mask"],
        "o2_right_arc_mask": common["right_arc_mask"],
        "a2_span_mask": common["angle_span_mask"],
        "remaining_mask": common["remaining_mask"],
        "uv_without_a2": common["uv_remaining"],
    }

    # uv is already expected to be outside A1 span.
    rel1 = uv - c1[None, :]
    theta1 = np.arctan2(rel1[:, 1], rel1[:, 0])
    phi = wrap_pi(theta1 - theta_top)
    cand_mask = np.ones(uv.shape[0], dtype=bool)

    left_mask = cand_mask & (phi > 0.0)
    right_mask = cand_mask & (phi < 0.0)

    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]
    p_left = uv[left_idx]
    p_right = uv[right_idx]

    if p_left.shape[0] < 3 or p_right.shape[0] < 3:
        raise RuntimeError("Not enough left/right points after O1 removal for A2 scan")

    # O2 residual inlier masks on side candidates
    res_l = np.abs(np.linalg.norm(p_left - c2l[None, :], axis=1) - float(o2_radius))
    res_r = np.abs(np.linalg.norm(p_right - c2r[None, :], axis=1) - float(o2_radius))
    in_l = res_l <= float(tol2)
    in_r = res_r <= float(tol2)

    # Beta from junction along downward direction on each O2 circle
    uj_l = (j_l - c2l) / float(o2_radius)
    uj_r = (j_r - c2r) / float(o2_radius)

    t1_l = np.array([-uj_l[1], uj_l[0]], dtype=np.float64)
    t2_l = -t1_l
    t_down_l = t1_l if t1_l[1] < t2_l[1] else t2_l
    dir_sign_l = np.sign(cross2d(uj_l, t_down_l))
    if dir_sign_l == 0:
        dir_sign_l = 1.0

    t1_r = np.array([-uj_r[1], uj_r[0]], dtype=np.float64)
    t2_r = -t1_r
    t_down_r = t1_r if t1_r[1] < t2_r[1] else t2_r
    dir_sign_r = np.sign(cross2d(uj_r, t_down_r))
    if dir_sign_r == 0:
        dir_sign_r = 1.0

    vec_l = p_left - c2l[None, :]
    vec_r = p_right - c2r[None, :]

    ang_l = np.arctan2(vec_l[:, 1], vec_l[:, 0])
    ang_r = np.arctan2(vec_r[:, 1], vec_r[:, 0])
    ang_j_l = np.arctan2((j_l - c2l)[1], (j_l - c2l)[0])
    ang_j_r = np.arctan2((j_r - c2r)[1], (j_r - c2r)[0])

    beta_l = dir_sign_l * wrap_pi(ang_l - ang_j_l)
    beta_r = dir_sign_r * wrap_pi(ang_r - ang_j_r)

    # keep only downward side progression
    beta_l_valid = beta_l >= 0.0
    beta_r_valid = beta_r >= 0.0

    step = np.deg2rad(step_deg)
    a2_max = np.deg2rad(max_a2_deg)
    edges = np.arange(0.0, a2_max + step, step, dtype=np.float64)
    if edges.shape[0] < 2:
        raise ValueError("No bins generated; check step_deg/max_a2_deg")
    beta_centers = 0.5 * (edges[:-1] + edges[1:])
    k = beta_centers.shape[0]

    left_ratio = np.zeros(k, dtype=np.float64)
    right_ratio = np.zeros(k, dtype=np.float64)
    left_count = np.zeros(k, dtype=np.int32)
    right_count = np.zeros(k, dtype=np.int32)
    accepted_bins = np.zeros(k, dtype=bool)

    accepted_until = -1
    consecutive_fail = 0

    for i in range(k):
        b0 = edges[i]
        b1 = edges[i + 1]

        lb = beta_l_valid & (beta_l >= b0) & (beta_l < b1)
        rb = beta_r_valid & (beta_r >= b0) & (beta_r < b1)

        lc = int(lb.sum())
        rc = int(rb.sum())
        left_count[i] = lc
        right_count[i] = rc

        lr = float(in_l[lb].mean()) if lc > 0 else 0.0
        rr = float(in_r[rb].mean()) if rc > 0 else 0.0
        left_ratio[i] = lr
        right_ratio[i] = rr

        ok_left = (lc >= min_bin_count) and (lr >= t_ratio)
        ok_right = (rc >= min_bin_count) and (rr >= t_ratio)
        ok = ok_left and ok_right
        accepted_bins[i] = ok

        if ok:
            accepted_until = i
            consecutive_fail = 0
        else:
            consecutive_fail += 1
            if consecutive_fail > fail_tolerance:
                break

    if accepted_until >= 0:
        a2_rad = float(edges[accepted_until + 1])
    else:
        a2_rad = 0.0
    a2_deg = float(np.rad2deg(a2_rad))

    # J2 junctions at A2 from J1 along O2 direction
    ang_j2_l = ang_j_l + dir_sign_l * a2_rad
    ang_j2_r = ang_j_r + dir_sign_r * a2_rad
    j2_left = c2l + float(o2_radius) * np.array([np.cos(ang_j2_l), np.sin(ang_j2_l)], dtype=np.float64)
    j2_right = c2r + float(o2_radius) * np.array([np.cos(ang_j2_r), np.sin(ang_j2_r)], dtype=np.float64)

    # Final masks in full uv indexing
    o2_left_arc_mask = np.zeros(uv.shape[0], dtype=bool)
    o2_right_arc_mask = np.zeros(uv.shape[0], dtype=bool)
    o2_left_arc_mask[left_idx] = beta_l_valid & (beta_l <= a2_rad) & in_l
    o2_right_arc_mask[right_idx] = beta_r_valid & (beta_r <= a2_rad) & in_r
    o2_arc_mask = o2_left_arc_mask | o2_right_arc_mask

    # Remove full A2 angular span (not only O2 inliers), for downstream O3 fitting.
    a2_left_span_mask = np.zeros(uv.shape[0], dtype=bool)
    a2_right_span_mask = np.zeros(uv.shape[0], dtype=bool)
    a2_left_span_mask[left_idx] = beta_l_valid & (beta_l <= a2_rad)
    a2_right_span_mask[right_idx] = beta_r_valid & (beta_r <= a2_rad)
    a2_span_mask = a2_left_span_mask | a2_right_span_mask
    remaining_mask = ~a2_span_mask
    uv_without_a2 = uv[remaining_mask]

    return {
        "A2_deg": a2_deg,
        "A2_rad": a2_rad,
        "j1_left": j_l,
        "j1_right": j_r,
        "j2_left": j2_left,
        "j2_right": j2_right,
        "accepted_until_bin": int(accepted_until),
        "beta_centers_deg": np.rad2deg(beta_centers),
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "left_count": left_count,
        "right_count": right_count,
        "accepted_bins": accepted_bins,
        "o2_arc_mask": o2_arc_mask,
        "o2_left_arc_mask": o2_left_arc_mask,
        "o2_right_arc_mask": o2_right_arc_mask,
        "a2_span_mask": a2_span_mask,
        "remaining_mask": remaining_mask,
        "uv_without_a2": uv_without_a2,
    }

def find_symmetric_a3_from_o3(
    uv_without_a2: np.ndarray,
    o1_center: np.ndarray,
    o1_radius: float,
    a1_deg: float,
    o2_left_center: np.ndarray,
    o2_right_center: np.ndarray,
    o2_radius: float,
    a2_deg: float,
    o3_left_center: np.ndarray,
    o3_right_center: np.ndarray,
    o3_radius: float,
    tol3: float = 0.1,
    t_ratio: float = 0.5,
    min_bin_count: int = 30,
    step_deg: float = 1.0,
    max_a3_deg: float = 140.0,
    fail_tolerance: int = 1,
):
    """
    Find symmetric A3 from O3 circles by ratio scan (same method as A1/A2).

    Returns:
        {
            "A3_deg", "A3_rad",
            "j2_left", "j2_right",
            "j3_left", "j3_right",
            "o3_left_span_mask", "o3_right_span_mask", "a3_span_mask",
            "remaining_mask", "uv_without_a3",
        }
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv_without_a2, dtype=np.float64)
    c1 = np.asarray(o1_center, dtype=np.float64).reshape(2)
    c2l = np.asarray(o2_left_center, dtype=np.float64).reshape(2)
    c2r = np.asarray(o2_right_center, dtype=np.float64).reshape(2)
    c3l = np.asarray(o3_left_center, dtype=np.float64).reshape(2)
    c3r = np.asarray(o3_right_center, dtype=np.float64).reshape(2)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv_without_a2 must have shape (N, 2)")
    if o1_radius <= 0 or o2_radius <= 0 or o3_radius <= 0:
        raise ValueError("o1_radius, o2_radius, o3_radius must be > 0")

    theta_top = np.pi / 2.0
    a1_rad = np.deg2rad(float(a1_deg))
    a2_rad = np.deg2rad(float(a2_deg))

    # Reconstruct J1
    th_l = theta_top + a1_rad
    th_r = theta_top - a1_rad
    n1_l = np.array([np.cos(th_l), np.sin(th_l)], dtype=np.float64)
    n1_r = np.array([np.cos(th_r), np.sin(th_r)], dtype=np.float64)
    j1_l = c1 + float(o1_radius) * n1_l
    j1_r = c1 + float(o1_radius) * n1_r

    # Reconstruct J2 from O2 + A2
    uj_l = (j1_l - c2l) / float(o2_radius)
    uj_r = (j1_r - c2r) / float(o2_radius)
    t1_l = np.array([-uj_l[1], uj_l[0]], dtype=np.float64)
    t2_l = -t1_l
    t1_r = np.array([-uj_r[1], uj_r[0]], dtype=np.float64)
    t2_r = -t1_r
    t_down_l = t1_l if t1_l[1] < t2_l[1] else t2_l
    t_down_r = t1_r if t1_r[1] < t2_r[1] else t2_r
    dir_sign_o2_l = np.sign(cross2d(uj_l, t_down_l))
    dir_sign_o2_r = np.sign(cross2d(uj_r, t_down_r))
    if dir_sign_o2_l == 0:
        dir_sign_o2_l = 1.0
    if dir_sign_o2_r == 0:
        dir_sign_o2_r = 1.0

    ang_j1_l = np.arctan2((j1_l - c2l)[1], (j1_l - c2l)[0])
    ang_j1_r = np.arctan2((j1_r - c2r)[1], (j1_r - c2r)[0])
    ang_j2_l = ang_j1_l + dir_sign_o2_l * a2_rad
    ang_j2_r = ang_j1_r + dir_sign_o2_r * a2_rad
    j2_l = c2l + float(o2_radius) * np.array([np.cos(ang_j2_l), np.sin(ang_j2_l)], dtype=np.float64)
    j2_r = c2r + float(o2_radius) * np.array([np.cos(ang_j2_r), np.sin(ang_j2_r)], dtype=np.float64)

    common = find_symmetric_angle_from_circles(
        uv=uv,
        side_split_center=c1,
        left_center=c3l,
        right_center=c3r,
        left_radius=float(o3_radius),
        right_radius=float(o3_radius),
        j_left=j2_l,
        j_right=j2_r,
        tol=float(tol3),
        t_ratio=t_ratio,
        min_bin_count=min_bin_count,
        step_deg=step_deg,
        max_angle_deg=max_a3_deg,
        fail_tolerance=fail_tolerance,
    )
    return {
        "A3_deg": common["angle_deg"],
        "A3_rad": common["angle_rad"],
        "j2_left": common["j_left"],
        "j2_right": common["j_right"],
        "j3_left": common["j_next_left"],
        "j3_right": common["j_next_right"],
        "o3_left_span_mask": common["left_arc_mask"],
        "o3_right_span_mask": common["right_arc_mask"],
        "a3_span_mask": common["angle_span_mask"],
        "remaining_mask": common["remaining_mask"],
        "uv_without_a3": common["uv_remaining"],
        "accepted_until_bin": common["accepted_until_bin"],
        "beta_centers_deg": common["bin_centers_deg"],
        "left_ratio": common["left_ratio"],
        "right_ratio": common["right_ratio"],
        "left_count": common["left_count"],
        "right_count": common["right_count"],
        "left_inlier_count": common["left_inlier_count"],
        "right_inlier_count": common["right_inlier_count"],
        "accepted_bins": common["accepted_bins"],
    }

def refine_o1_o2_o3_global_hard(
    uv: np.ndarray,
    o1_center_init: np.ndarray,
    r1_init: float,
    a1_deg_init: float,
    r2_init: float,
    a2_deg_init: float,
    r3_init: float,
    r1_bounds: tuple[float, float] | None = None,
    r2_bounds: tuple[float, float] | None = None,
    r3_bounds: tuple[float, float] | None = None,
    a1_max_deg: float = 170.0,
    a2_support_penalty_alpha: float = 0.05,
    support_bin_deg: float = 1.0,
    loss: str = "soft_l1",
    f_scale: float = 0.1,
    max_nfev: int = 300,
    verbose: int = 0,
):
    """
    Hard-constrained global refine (bounds-based).
    7-parameters : C1=(cx, cy), r1, A1, r2, A2, r3
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise ImportError("scipy is required for refine_o1_o2_o3_global_hard") from exc

    uv = np.asarray(uv, dtype=np.float64)
    c1_init = np.asarray(o1_center_init, dtype=np.float64).reshape(2)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if r1_init <= 0 or r2_init <= 0 or r3_init <= 0:
        raise ValueError("r1_init, r2_init, r3_init must be > 0")
    if not (1.0 < float(a1_max_deg) <= 170.0):
        raise ValueError("a1_max_deg must satisfy 1 < a1_max_deg <= 170")
    if float(a2_support_penalty_alpha) < 0.0:
        raise ValueError("a2_support_penalty_alpha must be >= 0")
    if float(support_bin_deg) <= 0.0:
        raise ValueError("support_bin_deg must be > 0")

    # 1) Angle bounds and numeric guard values.
    a1_min = np.deg2rad(1e-3)
    a1_max = np.deg2rad(float(a1_max_deg))
    a2_max = np.deg2rad(170.0)

    # 3) Build geometric model (centers/junctions/radii) from q.
    def build_model_q(q: np.ndarray):
        # q = [cx, cy, r1, r2, r3, a1, a2]
        cx, cy = float(q[0]), float(q[1])
        r1, r2, r3, a1, a2 = float(q[2]), float(q[3]), float(q[4]), float(q[5]), float(q[6])
        c1 = np.array([cx, cy], dtype=np.float64)
        theta_top = np.pi / 2.0

        th_l = theta_top + a1
        th_r = theta_top - a1
        n1_l = np.array([np.cos(th_l), np.sin(th_l)], dtype=np.float64)
        n1_r = np.array([np.cos(th_r), np.sin(th_r)], dtype=np.float64)
        j1_l = c1 + r1 * n1_l
        j1_r = c1 + r1 * n1_r
        c2_l = j1_l - n1_l * r2
        c2_r = j1_r - n1_r * r2
        ujl = (j1_l - c2_l) / r2
        ujr = (j1_r - c2_r) / r2
        t1_l = np.array([-ujl[1], ujl[0]], dtype=np.float64)
        t2_l = -t1_l
        t1_r = np.array([-ujr[1], ujr[0]], dtype=np.float64)
        t2_r = -t1_r
        t_down_l = t1_l if t1_l[1] < t2_l[1] else t2_l
        t_down_r = t1_r if t1_r[1] < t2_r[1] else t2_r
        dir_sign_l = np.sign(cross2d(ujl, t_down_l)) or 1.0
        dir_sign_r = np.sign(cross2d(ujr, t_down_r)) or 1.0
        ang_j1_l = np.arctan2((j1_l - c2_l)[1], (j1_l - c2_l)[0])
        ang_j1_r = np.arctan2((j1_r - c2_r)[1], (j1_r - c2_r)[0])
        ang_j2_l = ang_j1_l + dir_sign_l * a2
        ang_j2_r = ang_j1_r + dir_sign_r * a2
        j2_l = c2_l + r2 * np.array([np.cos(ang_j2_l), np.sin(ang_j2_l)], dtype=np.float64)
        j2_r = c2_r + r2 * np.array([np.cos(ang_j2_r), np.sin(ang_j2_r)], dtype=np.float64)
        n2_l = (j2_l - c2_l) / r2
        n2_r = (j2_r - c2_r) / r2
        c3_l = j2_l - n2_l * r3
        c3_r = j2_r - n2_r * r3

        return {
            "c1": c1,
            "r1": r1,
            "a1": a1,
            "c2_l": c2_l,
            "c2_r": c2_r,
            "r2": r2,
            "a2": a2,
            "j1_l": j1_l,
            "j1_r": j1_r,
            "j2_l": j2_l,
            "j2_r": j2_r,
            "c3_l": c3_l,
            "c3_r": c3_r,
            "r3": r3,
            "theta_top": theta_top,
        }

    def _unsupported_ratio(angle_vals: np.ndarray, max_angle: float, bin_deg: float) -> float:
        if max_angle <= 0.0:
            return 0.0
        n_bins = max(1, int(np.ceil(np.rad2deg(max_angle) / float(bin_deg))))
        edges = np.linspace(0.0, float(max_angle), n_bins + 1, dtype=np.float64)
        if angle_vals.size == 0:
            supported = 0
        else:
            hist, _ = np.histogram(angle_vals, bins=edges)
            supported = int(np.count_nonzero(hist > 0))
        unsupported = int(n_bins - supported)
        return float(unsupported) / float(max(supported, 1))

    # 4) Compute piecewise residual vector for least-squares.
    def point_residuals_q(q: np.ndarray):
        # 1) Decode unconstrained optimization variables q -> physical model parameters.
        #    build_model_q internally applies hard constraints (if enabled), then
        #    reconstructs O1/O2/O3 centers, radii and junction points.
        m = build_model_q(q)
        c1 = m["c1"]
        r1 = m["r1"]
        c2_l, c2_r, r2 = m["c2_l"], m["c2_r"], m["r2"]
        c3_l, c3_r, r3 = m["c3_l"], m["c3_r"], m["r3"]
        a1, a2 = m["a1"], m["a2"]
        theta_top = m["theta_top"]

        # 2) Global side split and O1 angular span.
        #    phi is angle relative to +v direction at C1:
        #      left  : phi > 0
        #      right : phi < 0
        #      O1    : |phi| <= A1
        rel1 = uv - c1[None, :]
        phi = wrap_pi(np.arctan2(rel1[:, 1], rel1[:, 0]) - theta_top)
        left = phi > 0.0
        right = phi < 0.0
        o1_span = np.abs(phi) <= a1

        # 3) Build per-side "downward progression" direction on O2 from J1 to J2.
        #    This sign is used to measure beta consistently as positive when moving
        #    downward along the side arc.
        ang_j1_l = np.arctan2((m["j1_l"] - c2_l)[1], (m["j1_l"] - c2_l)[0])
        ang_j1_r = np.arctan2((m["j1_r"] - c2_r)[1], (m["j1_r"] - c2_r)[0])
        dir_sign_l = np.sign(wrap_pi(np.array([np.arctan2((m["j2_l"] - c2_l)[1], (m["j2_l"] - c2_l)[0]) - ang_j1_l]))[0]) or 1.0
        dir_sign_r = np.sign(wrap_pi(np.array([np.arctan2((m["j2_r"] - c2_r)[1], (m["j2_r"] - c2_r)[0]) - ang_j1_r]))[0]) or 1.0

        # 4) Residual vector for least_squares (signed distance residuals).
        res = np.zeros(uv.shape[0], dtype=np.float64)

        # 4-1) O1 region residual: radial distance mismatch to O1 circle.
        d1 = np.linalg.norm(uv - c1[None, :], axis=1)
        res[o1_span] = d1[o1_span] - r1

        # 4-2) Left side points outside O1 span:
        #      - beta_l in [0, A2]  -> O2 residual
        #      - otherwise          -> O3 residual
        idx_l = np.where(left & (~o1_span))[0]
        if idx_l.size > 0:
            pts_l = uv[idx_l]
            vec_l = pts_l - c2_l[None, :]
            ang_l = np.arctan2(vec_l[:, 1], vec_l[:, 0])
            beta_l = dir_sign_l * wrap_pi(ang_l - ang_j1_l)
            beta_l_valid = beta_l >= 0.0
            o2_l = beta_l_valid & (beta_l <= a2)
            if np.any(o2_l):
                pts = pts_l[o2_l]
                res[idx_l[o2_l]] = np.linalg.norm(pts - c2_l[None, :], axis=1) - r2
            o3_l = beta_l_valid & (~o2_l)
            if np.any(o3_l):
                pts = pts_l[o3_l]
                res[idx_l[o3_l]] = np.linalg.norm(pts - c3_l[None, :], axis=1) - r3

        # 4-3) Right side points outside O1 span:
        #      same rule as left side with right-center geometry.
        idx_r = np.where(right & (~o1_span))[0]
        if idx_r.size > 0:
            pts_r = uv[idx_r]
            vec_r = pts_r - c2_r[None, :]
            ang_r = np.arctan2(vec_r[:, 1], vec_r[:, 0])
            beta_r = dir_sign_r * wrap_pi(ang_r - ang_j1_r)
            beta_r_valid = beta_r >= 0.0
            o2_r = beta_r_valid & (beta_r <= a2)
            if np.any(o2_r):
                pts = pts_r[o2_r]
                res[idx_r[o2_r]] = np.linalg.norm(pts - c2_r[None, :], axis=1) - r2
            o3_r = beta_r_valid & (~o2_r)
            if np.any(o3_r):
                pts = pts_r[o3_r]
                res[idx_r[o3_r]] = np.linalg.norm(pts - c3_r[None, :], axis=1) - r3

        # 4-4) Points exactly on split centerline (phi==0) and outside O1:
        #      assign to the nearer O3 residual to avoid undefined left/right choice.
        centerline_idx = np.where((~left) & (~right) & (~o1_span))[0]
        if centerline_idx.size > 0:
            pts = uv[centerline_idx]
            dl = np.linalg.norm(pts - c3_l[None, :], axis=1) - r3
            dr = np.linalg.norm(pts - c3_r[None, :], axis=1) - r3
            choose_left = np.abs(dl) <= np.abs(dr)
            res_center = np.empty(centerline_idx.size, dtype=np.float64)
            res_center[choose_left] = dl[choose_left]
            res_center[~choose_left] = dr[~choose_left]
            res[centerline_idx] = res_center

        # A2 support penalty (5-circle): penalize unsupported O2 tail length.
        if float(a2_support_penalty_alpha) > 0.0:
            beta_l_for_pen = np.empty((0,), dtype=np.float64)
            beta_r_for_pen = np.empty((0,), dtype=np.float64)
            if idx_l.size > 0:
                beta_l_for_pen = beta_l[beta_l_valid & (beta_l <= a2)]
            if idx_r.size > 0:
                beta_r_for_pen = beta_r[beta_r_valid & (beta_r <= a2)]
            beta_all = np.concatenate((beta_l_for_pen, beta_r_for_pen), axis=0)
            ratio = _unsupported_ratio(beta_all, float(a2), float(support_bin_deg))
            n_eff = max(int(beta_all.size), 1)
            pen = float(a2_support_penalty_alpha) * np.sqrt(float(n_eff)) * ratio
            res = np.concatenate((res, np.array([pen], dtype=np.float64)), axis=0)

        # least_squares minimizes sum(residual_i^2) (with optional robust loss).
        return res

    def _compute_a2_unsupported_ratio(m: dict) -> float:
        c1 = m["c1"]
        c2_l, c2_r = m["c2_l"], m["c2_r"]
        a1, a2 = float(m["a1"]), float(m["a2"])
        theta_top = m["theta_top"]

        rel1 = uv - c1[None, :]
        phi = wrap_pi(np.arctan2(rel1[:, 1], rel1[:, 0]) - theta_top)
        left = phi > 0.0
        right = phi < 0.0
        o1_span = np.abs(phi) <= a1

        ang_j1_l = np.arctan2((m["j1_l"] - c2_l)[1], (m["j1_l"] - c2_l)[0])
        ang_j1_r = np.arctan2((m["j1_r"] - c2_r)[1], (m["j1_r"] - c2_r)[0])
        dir_sign_l = np.sign(
            wrap_pi(
                np.array(
                    [np.arctan2((m["j2_l"] - c2_l)[1], (m["j2_l"] - c2_l)[0]) - ang_j1_l]
                )
            )[0]
        ) or 1.0
        dir_sign_r = np.sign(
            wrap_pi(
                np.array(
                    [np.arctan2((m["j2_r"] - c2_r)[1], (m["j2_r"] - c2_r)[0]) - ang_j1_r]
                )
            )[0]
        ) or 1.0

        idx_l = np.where(left & (~o1_span))[0]
        idx_r = np.where(right & (~o1_span))[0]
        beta_l_for_pen = np.empty((0,), dtype=np.float64)
        beta_r_for_pen = np.empty((0,), dtype=np.float64)
        if idx_l.size > 0:
            pts_l = uv[idx_l]
            vec_l = pts_l - c2_l[None, :]
            ang_l = np.arctan2(vec_l[:, 1], vec_l[:, 0])
            beta_l = dir_sign_l * wrap_pi(ang_l - ang_j1_l)
            beta_l_valid = beta_l >= 0.0
            beta_l_for_pen = beta_l[beta_l_valid & (beta_l <= a2)]
        if idx_r.size > 0:
            pts_r = uv[idx_r]
            vec_r = pts_r - c2_r[None, :]
            ang_r = np.arctan2(vec_r[:, 1], vec_r[:, 0])
            beta_r = dir_sign_r * wrap_pi(ang_r - ang_j1_r)
            beta_r_valid = beta_r >= 0.0
            beta_r_for_pen = beta_r[beta_r_valid & (beta_r <= a2)]
        beta_all = np.concatenate((beta_l_for_pen, beta_r_for_pen), axis=0)
        return float(_unsupported_ratio(beta_all, float(a2), float(support_bin_deg)))

    # 5) Build initial optimizer vector q0 in physical parameter space.
    r1_0 = max(float(r1_init), 1e-6)
    r2_0 = max(float(r2_init), 1e-6)
    r3_0 = max(float(r3_init), 1e-6)
    a1_0 = np.clip(np.deg2rad(float(a1_deg_init)), a1_min, a1_max)
    a2_0 = np.clip(np.deg2rad(float(a2_deg_init)), a1_min, a2_max)
    q0 = np.array(
        [
            float(c1_init[0]),
            float(c1_init[1]),
            float(r1_0),
            float(r2_0),
            float(r3_0),
            float(a1_0),
            float(a2_0),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(q0)):
        raise ValueError("Initial guess contains non-finite values")
    if r1_bounds is None:
        r1_lo, r1_hi = 1e-6, np.inf
    else:
        r1_lo, r1_hi = float(r1_bounds[0]), float(r1_bounds[1])
    if r2_bounds is None:
        r2_lo, r2_hi = 1e-6, np.inf
    else:
        r2_lo, r2_hi = float(r2_bounds[0]), float(r2_bounds[1])
    if r3_bounds is None:
        r3_lo, r3_hi = 1e-6, np.inf
    else:
        r3_lo, r3_hi = float(r3_bounds[0]), float(r3_bounds[1])
    if r1_lo < 0.0 or r2_lo < 0.0 or r3_lo < 0.0:
        raise ValueError("r1/r2/r3 bounds lower must be >= 0")
    if r1_hi <= r1_lo or r2_hi <= r2_lo or r3_hi <= r3_lo:
        raise ValueError("r1/r2/r3 bounds upper must be > lower")
    q0[2] = float(np.clip(q0[2], r1_lo + 1e-12, r1_hi - 1e-12 if np.isfinite(r1_hi) else q0[2]))
    q0[3] = float(np.clip(q0[3], r2_lo + 1e-12, r2_hi - 1e-12 if np.isfinite(r2_hi) else q0[3]))
    q0[4] = float(np.clip(q0[4], r3_lo + 1e-12, r3_hi - 1e-12 if np.isfinite(r3_hi) else q0[4]))
    lb = np.array([-np.inf, -np.inf, r1_lo, r2_lo, r3_lo, a1_min, a1_min], dtype=np.float64)
    ub = np.array([np.inf, np.inf, r1_hi, r2_hi, r3_hi, a1_max, a2_max], dtype=np.float64)

    # 6) Run robust least-squares optimization.
    opt = least_squares(
        point_residuals_q,
        q0,
        bounds=(lb, ub),
        loss=loss,
        f_scale=float(f_scale),
        max_nfev=int(max_nfev),
        verbose=int(verbose),
    )

    # 7) Decode optimized parameters and return standardized output.
    m = build_model_q(opt.x)
    return {
        "success": bool(opt.success),
        "status": int(opt.status),
        "message": str(opt.message),
        "cost": float(opt.cost),
        "nfev": int(opt.nfev),
        "x_opt": opt.x,
        "o1_center": m["c1"],
        "r1": float(m["r1"]),
        "a1_deg": float(np.rad2deg(m["a1"])),
        "o2_left_center": m["c2_l"],
        "o2_right_center": m["c2_r"],
        "r2": float(m["r2"]),
        "a2_deg": float(np.rad2deg(m["a2"])),
        "o3_left_center": m["c3_l"],
        "o3_right_center": m["c3_r"],
        "r3": float(m["r3"]),
        "j1_left": m["j1_l"],
        "j1_right": m["j1_r"],
        "j2_left": m["j2_l"],
        "j2_right": m["j2_r"],
        "a2_unsupported_ratio": float(_compute_a2_unsupported_ratio(m)),
    }

def run_five_circle_fit(
    uv: np.ndarray,
    *,
    init_top_range: float = 50.0,
    a1_max_deg_init: float = 90.0,
    init_r_lo_scale: float = 0.2,
    init_r_hi_scale: float = 3.0,
    o1_residual_threshold: float = 0.1,
    o1_max_trials: int = 500,
    o1_min_inliers: int = 1,
    o2_residual_threshold: float = 0.2,
    o2_max_trials: int = 50,
    o2_min_inliers_per_side: int = 1,
    o2_refit: bool = False,
    o3_residual_threshold: float = 0.2,
    o3_max_trials: int = 50,
    o3_min_inliers_per_side: int = 1,
    o3_refit: bool = False,
    a1_init_residual_threshold: float = 0.1,
    a1_init_ratio: float = 0.2,
    a1_init_fail_tolerance: int = 3,
    a2_init_residual_threshold: float = 0.1,
    a2_init_ratio: float = 0.2,
    a2_init_fail_tolerance: int = 5,
    a_final_residual_threshold: float = 0.5,
    a_final_ratio: float = 0.1,
    a_final_step_deg: float = 0.1,
    a_final_fail_tolerance: int = 10,
    loss: str = "soft_l1",
    f_scale: float = 0.1,
    max_nfev: int = 300,
    refine_a1_max_deg: float = 170.0,
    refine_r_lo_scale: float = 0.1,
    refine_r_hi_scale: float = 5.0,
    fallback_a_init_shrink: float = 0.5,
    refine_a2_penalty_alpha_5: float = 0.05,
    refine_support_bin_deg: float = 1.0,
    random_seed: int | None = 0,
    **_unused,
) -> dict:
    """Run only the 5-circle pipeline used by PyVITA."""
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if uv.shape[0] < 3:
        raise ValueError("Need at least 3 points")

    u_range = float(np.max(uv[:, 0]) - np.min(uv[:, 0]))
    v_range = float(np.max(uv[:, 1]) - np.min(uv[:, 1]))
    r_base = float(max(u_range, v_range))
    r_lo_search = float(r_base * float(init_r_lo_scale)) or 1e-6
    r_hi_search = float(r_base * float(init_r_hi_scale))
    r_lo_refine = float(r_base * float(refine_r_lo_scale)) or 1e-6
    r_hi_refine = float(r_base * float(refine_r_hi_scale))

    def _clip_uv_for_init(src_uv):
        return clip_uv_top_percent_by_z_and_plot(
            src_uv,
            top_range=float(init_top_range),
            z_col=1,
        )["uv_clipped"]

    fit_o1 = fit_circle_ransac(
        uv=_clip_uv_for_init(uv),
        residual_threshold=float(o1_residual_threshold),
        max_trials=int(o1_max_trials),
        min_inliers=int(o1_min_inliers),
        random_seed=random_seed,
        refit=True,
    )
    a1 = find_symmetric_a1_from_o1(
        uv=uv,
        center=fit_o1["center"],
        radius=float(fit_o1["radius"]),
        tol1=float(a1_init_residual_threshold),
        t_ratio=float(a1_init_ratio),
        min_bin_count=1,
        step_deg=1.0,
        max_alpha_deg=float(a1_max_deg_init),
        fail_tolerance=int(a1_init_fail_tolerance),
    )
    fit_o2 = fit_o2_symmetric_constrained_ransac(
        uv_without_a1=_clip_uv_for_init(a1["uv_without_a1"]),
        o1_center=fit_o1["center"],
        o1_radius=float(fit_o1["radius"]),
        a1_deg=float(a1["A1_deg"]),
        residual_threshold=float(o2_residual_threshold),
        max_trials=int(o2_max_trials),
        min_inliers_per_side=int(o2_min_inliers_per_side),
        random_seed=random_seed,
        refit=bool(o2_refit),
        enforce_c2_between_c1_and_j=False,
        r_min=r_lo_search,
        r_hi=r_hi_search,
    )
    a2 = find_symmetric_a2_from_o2(
        uv_without_a1=a1["uv_without_a1"],
        o1_center=fit_o1["center"],
        o1_radius=float(fit_o1["radius"]),
        a1_deg=float(a1["A1_deg"]),
        o2_left_center=fit_o2["o2_left_center"],
        o2_right_center=fit_o2["o2_right_center"],
        o2_radius=float(fit_o2["r2"]),
        tol2=float(a2_init_residual_threshold),
        t_ratio=float(a2_init_ratio),
        min_bin_count=1,
        step_deg=1.0,
        max_a2_deg=170.0,
        fail_tolerance=int(a2_init_fail_tolerance),
    )
    fit_o3 = fit_o3_symmetric_constrained_ransac(
        uv_without_a2=a2["uv_without_a2"],
        o1_center=fit_o1["center"],
        o1_radius=float(fit_o1["radius"]),
        a1_deg=float(a1["A1_deg"]),
        o2_left_center=fit_o2["o2_left_center"],
        o2_right_center=fit_o2["o2_right_center"],
        o2_radius=float(fit_o2["r2"]),
        a2_deg=float(a2["A2_deg"]),
        residual_threshold=float(o3_residual_threshold),
        max_trials=int(o3_max_trials),
        min_inliers_per_side=int(o3_min_inliers_per_side),
        random_seed=random_seed,
        refit=bool(o3_refit),
        enforce_r3_gt_r2=False,
        r_min=r_lo_search,
        r_hi=r_hi_search,
    )

    a2_refine_init = float(a2["A2_deg"])
    if bool(fit_o3.get("fallback_used", False)):
        a2_refine_init = max(1e-3, a2_refine_init * float(fallback_a_init_shrink))

    refined = refine_o1_o2_o3_global_hard(
        uv=uv,
        o1_center_init=fit_o1["center"],
        r1_init=float(fit_o1["radius"]),
        a1_deg_init=float(a1["A1_deg"]),
        r2_init=float(fit_o2["r2"]),
        a2_deg_init=float(a2_refine_init),
        r3_init=float(fit_o3["r3"]),
        r1_bounds=(float(r_lo_refine), float(r_hi_refine)),
        r2_bounds=(float(r_lo_refine), float(r_hi_refine)),
        r3_bounds=(float(r_lo_refine), float(r_hi_refine)),
        a1_max_deg=float(refine_a1_max_deg),
        a2_support_penalty_alpha=float(refine_a2_penalty_alpha_5),
        support_bin_deg=float(refine_support_bin_deg),
        loss=loss,
        f_scale=float(f_scale),
        max_nfev=int(max_nfev),
        verbose=0,
    )

    acc = compute_piecewise_circle_residuals(
        uv=uv,
        o1_center=refined["o1_center"],
        o1_radius=float(refined["r1"]),
        a1_deg=float(refined["a1_deg"]),
        o2_left_center=refined["o2_left_center"],
        o2_right_center=refined["o2_right_center"],
        o2_radius=float(refined["r2"]),
        a2_deg=float(refined["a2_deg"]),
        o3_left_center=refined["o3_left_center"],
        o3_right_center=refined["o3_right_center"],
        o3_radius=float(refined["r3"]),
    )
    uv3_ref = uv[~(acc["o1_mask"] | acc["o2_mask"])]
    a3 = find_symmetric_a3_from_o3(
        uv_without_a2=uv3_ref,
        o1_center=refined["o1_center"],
        o1_radius=float(refined["r1"]),
        a1_deg=float(refined["a1_deg"]),
        o2_left_center=refined["o2_left_center"],
        o2_right_center=refined["o2_right_center"],
        o2_radius=float(refined["r2"]),
        a2_deg=float(refined["a2_deg"]),
        o3_left_center=refined["o3_left_center"],
        o3_right_center=refined["o3_right_center"],
        o3_radius=float(refined["r3"]),
        tol3=float(a_final_residual_threshold),
        t_ratio=float(a_final_ratio),
        min_bin_count=1,
        step_deg=float(a_final_step_deg),
        max_a3_deg=170.0,
        fail_tolerance=int(a_final_fail_tolerance),
    )
    return {
        "ok": True,
        "initial_fit": {
            "o1_center": fit_o1["center"],
            "r1": float(fit_o1["radius"]),
            "a1_deg": float(a1["A1_deg"]),
            "o2_left_center": fit_o2["o2_left_center"],
            "o2_right_center": fit_o2["o2_right_center"],
            "r2": float(fit_o2["r2"]),
            "a2_deg": float(a2["A2_deg"]),
            "o3_left_center": fit_o3["o3_left_center"],
            "o3_right_center": fit_o3["o3_right_center"],
            "r3": float(fit_o3["r3"]),
        },
        "refined": refined,
        "final_angles": {"a3_deg": float(a3["A3_deg"])},
        "accuracy": acc,
    }

def build_o1_o2_o3_profile_polyline(
    o1_center: np.ndarray,
    o1_radius: float,
    a1_deg: float,
    o2_left_center: np.ndarray,
    o2_right_center: np.ndarray,
    o2_radius: float,
    a2_deg: float,
    o3_left_center: np.ndarray,
    o3_right_center: np.ndarray,
    o3_radius: float,
    a3_deg: float,
    n_per_arc: int = 200,
    close_polygon: bool = False,
) -> dict[str, np.ndarray]:
    """
    Build final 3-center tunnel profile polyline from O1/O2/O3 arcs.

    Arc order (continuous):
    O3L(J3->J2) -> O2L(J2->J1) -> O1(J1L->J1R) -> O2R(J1->J2) -> O3R(J2->J3)
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    c1 = np.asarray(o1_center, dtype=np.float64).reshape(2)
    c2l = np.asarray(o2_left_center, dtype=np.float64).reshape(2)
    c2r = np.asarray(o2_right_center, dtype=np.float64).reshape(2)
    c3l = np.asarray(o3_left_center, dtype=np.float64).reshape(2)
    c3r = np.asarray(o3_right_center, dtype=np.float64).reshape(2)

    if min(float(o1_radius), float(o2_radius), float(o3_radius)) <= 0.0:
        raise ValueError("all radii must be > 0")
    if n_per_arc < 8:
        raise ValueError("n_per_arc must be >= 8")

    def down_sign_at_junction(center: np.ndarray, junction: np.ndarray) -> float:
        u = junction - center
        nu = float(np.linalg.norm(u))
        if nu <= 1e-12:
            raise ValueError("invalid center/junction geometry")
        u = u / nu
        t1 = np.array([-u[1], u[0]], dtype=np.float64)
        t2 = -t1
        t_down = t1 if t1[1] < t2[1] else t2
        s = np.sign(cross2d(u, t_down))
        return float(s if s != 0 else 1.0)

    a1 = np.deg2rad(float(a1_deg))
    a2 = np.deg2rad(float(a2_deg))
    a3 = np.deg2rad(float(a3_deg))
    theta_top = np.pi / 2.0

    # J1 from O1
    th_j1_l = theta_top + a1
    th_j1_r = theta_top - a1
    j1_l = c1 + float(o1_radius) * np.array([np.cos(th_j1_l), np.sin(th_j1_l)], dtype=np.float64)
    j1_r = c1 + float(o1_radius) * np.array([np.cos(th_j1_r), np.sin(th_j1_r)], dtype=np.float64)

    # J2 from O2 + A2
    s2_l = down_sign_at_junction(c2l, j1_l)
    s2_r = down_sign_at_junction(c2r, j1_r)
    ang_j1_l = np.arctan2((j1_l - c2l)[1], (j1_l - c2l)[0])
    ang_j1_r = np.arctan2((j1_r - c2r)[1], (j1_r - c2r)[0])
    ang_j2_l = ang_j1_l + s2_l * a2
    ang_j2_r = ang_j1_r + s2_r * a2
    j2_l = c2l + float(o2_radius) * np.array([np.cos(ang_j2_l), np.sin(ang_j2_l)], dtype=np.float64)
    j2_r = c2r + float(o2_radius) * np.array([np.cos(ang_j2_r), np.sin(ang_j2_r)], dtype=np.float64)

    # J3 from O3 + A3
    s3_l = down_sign_at_junction(c3l, j2_l)
    s3_r = down_sign_at_junction(c3r, j2_r)
    ang_j3_l = np.arctan2((j2_l - c3l)[1], (j2_l - c3l)[0]) + s3_l * a3
    ang_j3_r = np.arctan2((j2_r - c3r)[1], (j2_r - c3r)[0]) + s3_r * a3
    j3_l = c3l + float(o3_radius) * np.array([np.cos(ang_j3_l), np.sin(ang_j3_l)], dtype=np.float64)
    j3_r = c3r + float(o3_radius) * np.array([np.cos(ang_j3_r), np.sin(ang_j3_r)], dtype=np.float64)

    # Arc samples
    t = np.linspace(0.0, 1.0, int(n_per_arc), dtype=np.float64)

    # O3L: J3 -> J2
    a_l_o3 = np.arctan2((j2_l - c3l)[1], (j2_l - c3l)[0]) + s3_l * (a3 * (1.0 - t))
    arc_o3l = c3l[None, :] + float(o3_radius) * np.column_stack((np.cos(a_l_o3), np.sin(a_l_o3)))

    # O2L: J2 -> J1
    a_l_o2 = ang_j1_l + s2_l * (a2 * (1.0 - t))
    arc_o2l = c2l[None, :] + float(o2_radius) * np.column_stack((np.cos(a_l_o2), np.sin(a_l_o2)))

    # O1: J1L -> J1R across crown
    a_o1 = (theta_top + a1) + (-2.0 * a1) * t
    arc_o1 = c1[None, :] + float(o1_radius) * np.column_stack((np.cos(a_o1), np.sin(a_o1)))

    # O2R: J1 -> J2
    a_r_o2 = ang_j1_r + s2_r * (a2 * t)
    arc_o2r = c2r[None, :] + float(o2_radius) * np.column_stack((np.cos(a_r_o2), np.sin(a_r_o2)))

    # O3R: J2 -> J3
    a_r_o3 = np.arctan2((j2_r - c3r)[1], (j2_r - c3r)[0]) + s3_r * (a3 * t)
    arc_o3r = c3r[None, :] + float(o3_radius) * np.column_stack((np.cos(a_r_o3), np.sin(a_r_o3)))

    polyline = np.vstack((arc_o3l, arc_o2l[1:], arc_o1[1:], arc_o2r[1:], arc_o3r[1:]))
    if close_polygon:
        polyline = np.vstack((polyline, polyline[:1]))

    return {
        "polyline": polyline,
        "arc_o3_left": arc_o3l,
        "arc_o2_left": arc_o2l,
        "arc_o1": arc_o1,
        "arc_o2_right": arc_o2r,
        "arc_o3_right": arc_o3r,
        "j1_left": j1_l,
        "j1_right": j1_r,
        "j2_left": j2_l,
        "j2_right": j2_r,
        "j3_left": j3_l,
        "j3_right": j3_r,
    }

def compute_piecewise_circle_residuals(
    uv: np.ndarray,
    o1_center: np.ndarray,
    o1_radius: float,
    a1_deg: float | None = None,
    o2_left_center: np.ndarray | None = None,
    o2_right_center: np.ndarray | None = None,
    o2_radius: float | None = None,
    a2_deg: float | None = None,
    o3_left_center: np.ndarray | None = None,
    o3_right_center: np.ndarray | None = None,
    o3_radius: float | None = None,
) -> dict[str, np.ndarray | float]:
    """
    Compute residuals for 5/3/1-circle models with unified interface.

    Auto model selection by provided arguments:
    - 5-circle: o2*, a2_deg, o3* are all provided
    - 3-circle: o2* provided, o3* not provided
    - 1-circle: only O1 provided

    Angle requirements by mode:
    - 1-circle: a1_deg not required
    - 3-circle: a1_deg required
    - 5-circle: a1_deg and a2_deg required
    """
    # 1) Validate inputs and normalize arrays.
    # 2) Run the core computation for this function.
    # 3) Build and return the output structure.
    uv = np.asarray(uv, dtype=np.float64)
    c1 = np.asarray(o1_center, dtype=np.float64).reshape(2)
    r1 = float(o1_radius)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if r1 <= 0:
        raise ValueError("o1_radius must be > 0")

    has_o2 = (o2_left_center is not None) and (o2_right_center is not None) and (o2_radius is not None)
    has_o3 = (o3_left_center is not None) and (o3_right_center is not None) and (o3_radius is not None)
    if has_o3 and not has_o2:
        raise ValueError("O3 is provided but O2 is missing")
    if has_o3 and a2_deg is None:
        raise ValueError("a2_deg is required when O3 is provided (5-circle mode)")

    theta_top = np.pi / 2.0
    rel1 = uv - c1[None, :]
    phi = wrap_pi(np.arctan2(rel1[:, 1], rel1[:, 0]) - theta_top)
    left_side = phi > 0.0
    right_side = phi < 0.0

    residuals = np.full(uv.shape[0], np.nan, dtype=np.float64)
    o1_mask = np.zeros(uv.shape[0], dtype=bool)

    # Default masks (for consistent return keys in all model modes)
    o2_left_mask = np.zeros(uv.shape[0], dtype=bool)
    o2_right_mask = np.zeros(uv.shape[0], dtype=bool)
    o2_mask = np.zeros(uv.shape[0], dtype=bool)
    o3_mask = np.zeros(uv.shape[0], dtype=bool)
    o3_left_mask = np.zeros(uv.shape[0], dtype=bool)
    o3_right_mask = np.zeros(uv.shape[0], dtype=bool)
    centerline_mask = np.zeros(uv.shape[0], dtype=bool)

    # model mode decision
    if not has_o2:
        model_mode = "1circle"
    elif not has_o3:
        model_mode = "3circle"
    else:
        model_mode = "5circle"

    if model_mode == "1circle":
        # Single-circle residual for all points.
        residuals[:] = np.abs(np.linalg.norm(uv - c1[None, :], axis=1) - r1)
        o1_mask[:] = True
        if a1_deg is not None:
            a1 = np.deg2rad(float(a1_deg))
            j1_l = c1 + r1 * np.array([np.cos(theta_top + a1), np.sin(theta_top + a1)], dtype=np.float64)
            j1_r = c1 + r1 * np.array([np.cos(theta_top - a1), np.sin(theta_top - a1)], dtype=np.float64)
        else:
            j1_l = np.array([np.nan, np.nan], dtype=np.float64)
            j1_r = np.array([np.nan, np.nan], dtype=np.float64)
        j2_l = j1_l.copy()
        j2_r = j1_r.copy()
    else:
        if a1_deg is None:
            raise ValueError("a1_deg is required for 3-circle/5-circle modes")
        a1 = np.deg2rad(float(a1_deg))
        o1_mask = np.abs(phi) <= a1
        c2l = np.asarray(o2_left_center, dtype=np.float64).reshape(2)
        c2r = np.asarray(o2_right_center, dtype=np.float64).reshape(2)
        r2 = float(o2_radius)
        if r2 <= 0:
            raise ValueError("o2_radius must be > 0")

        # O1 stage
        if np.any(o1_mask):
            residuals[o1_mask] = np.abs(np.linalg.norm(uv[o1_mask] - c1[None, :], axis=1) - r1)

        # J1 from O1
        th_l = theta_top + a1
        th_r = theta_top - a1
        j1_l = c1 + r1 * np.array([np.cos(th_l), np.sin(th_l)], dtype=np.float64)
        j1_r = c1 + r1 * np.array([np.cos(th_r), np.sin(th_r)], dtype=np.float64)

        if model_mode == "3circle":
            # Stage2 for all non-O1 points (no A2 split, no O3).
            o2_left_mask = left_side & (~o1_mask)
            o2_right_mask = right_side & (~o1_mask)
            o2_mask = o2_left_mask | o2_right_mask
            if np.any(o2_left_mask):
                residuals[o2_left_mask] = np.abs(np.linalg.norm(uv[o2_left_mask] - c2l[None, :], axis=1) - r2)
            if np.any(o2_right_mask):
                residuals[o2_right_mask] = np.abs(np.linalg.norm(uv[o2_right_mask] - c2r[None, :], axis=1) - r2)
            centerline_mask = (~left_side) & (~right_side) & (~o1_mask)
            if np.any(centerline_mask):
                pts = uv[centerline_mask]
                dl = np.abs(np.linalg.norm(pts - c2l[None, :], axis=1) - r2)
                dr = np.abs(np.linalg.norm(pts - c2r[None, :], axis=1) - r2)
                residuals[centerline_mask] = np.minimum(dl, dr)
            j2_l = j1_l.copy()
            j2_r = j1_r.copy()
        else:
            # 5-circle mode
            c3l = np.asarray(o3_left_center, dtype=np.float64).reshape(2)
            c3r = np.asarray(o3_right_center, dtype=np.float64).reshape(2)
            r3 = float(o3_radius)
            if r3 <= 0:
                raise ValueError("o3_radius must be > 0")
            a2 = np.deg2rad(float(a2_deg))

            # O2 direction and J2
            uj_l = (j1_l - c2l) / r2
            uj_r = (j1_r - c2r) / r2
            t1_l = np.array([-uj_l[1], uj_l[0]], dtype=np.float64)
            t2_l = -t1_l
            t1_r = np.array([-uj_r[1], uj_r[0]], dtype=np.float64)
            t2_r = -t1_r
            t_down_l = t1_l if t1_l[1] < t2_l[1] else t2_l
            t_down_r = t1_r if t1_r[1] < t2_r[1] else t2_r
            dir2_l = np.sign(cross2d(uj_l, t_down_l)) or 1.0
            dir2_r = np.sign(cross2d(uj_r, t_down_r)) or 1.0
            ang_j1_l = np.arctan2((j1_l - c2l)[1], (j1_l - c2l)[0])
            ang_j1_r = np.arctan2((j1_r - c2r)[1], (j1_r - c2r)[0])
            ang_j2_l = ang_j1_l + dir2_l * a2
            ang_j2_r = ang_j1_r + dir2_r * a2
            j2_l = c2l + r2 * np.array([np.cos(ang_j2_l), np.sin(ang_j2_l)], dtype=np.float64)
            j2_r = c2r + r2 * np.array([np.cos(ang_j2_r), np.sin(ang_j2_r)], dtype=np.float64)

            # O2 stage on non-O1 points.
            idx_l = np.where(left_side & (~o1_mask))[0]
            idx_r = np.where(right_side & (~o1_mask))[0]
            if idx_l.size > 0:
                p = uv[idx_l]
                ang = np.arctan2((p - c2l[None, :])[:, 1], (p - c2l[None, :])[:, 0])
                beta = dir2_l * wrap_pi(ang - ang_j1_l)
                m = (beta >= 0.0) & (beta <= a2)
                o2_left_mask[idx_l[m]] = True
            if idx_r.size > 0:
                p = uv[idx_r]
                ang = np.arctan2((p - c2r[None, :])[:, 1], (p - c2r[None, :])[:, 0])
                beta = dir2_r * wrap_pi(ang - ang_j1_r)
                m = (beta >= 0.0) & (beta <= a2)
                o2_right_mask[idx_r[m]] = True
            o2_mask = o2_left_mask | o2_right_mask
            if np.any(o2_left_mask):
                residuals[o2_left_mask] = np.abs(np.linalg.norm(uv[o2_left_mask] - c2l[None, :], axis=1) - r2)
            if np.any(o2_right_mask):
                residuals[o2_right_mask] = np.abs(np.linalg.norm(uv[o2_right_mask] - c2r[None, :], axis=1) - r2)

            # O3 stage for all remaining points.
            o3_mask = ~(o1_mask | o2_mask)
            o3_left_mask = o3_mask & left_side
            o3_right_mask = o3_mask & right_side
            centerline_mask = o3_mask & (~left_side) & (~right_side)
            if np.any(o3_left_mask):
                residuals[o3_left_mask] = np.abs(np.linalg.norm(uv[o3_left_mask] - c3l[None, :], axis=1) - r3)
            if np.any(o3_right_mask):
                residuals[o3_right_mask] = np.abs(np.linalg.norm(uv[o3_right_mask] - c3r[None, :], axis=1) - r3)
            if np.any(centerline_mask):
                pts = uv[centerline_mask]
                dl = np.abs(np.linalg.norm(pts - c3l[None, :], axis=1) - r3)
                dr = np.abs(np.linalg.norm(pts - c3r[None, :], axis=1) - r3)
                residuals[centerline_mask] = np.minimum(dl, dr)

    valid = np.isfinite(residuals)
    if not np.any(valid):
        raise RuntimeError("No valid residuals were assigned")
    d = residuals[valid]

    return {
        "residuals": residuals,
        "valid_mask": valid,
        "o1_mask": o1_mask,
        "o2_mask": o2_mask,
        "o2_left_mask": o2_left_mask,
        "o2_right_mask": o2_right_mask,
        "o3_mask": o3_mask,
        "o3_left_mask": o3_left_mask,
        "o3_right_mask": o3_right_mask,
        "centerline_mask": centerline_mask,
        "left_side_mask": left_side,
        "right_side_mask": right_side,
        "j1_left": j1_l,
        "j1_right": j1_r,
        "j2_left": j2_l,
        "j2_right": j2_r,
        "model_mode": model_mode,
        "mae": float(np.mean(d)),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "p50": float(np.percentile(d, 50)),
        "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)),
        "max": float(np.max(d)),
        "count": int(d.shape[0]),
    }
