"""Adapter for the 5-circle tunnel profile fitter."""
import json
import os
import numpy as np


DEFAULT_FIVE_CIRCLE_PARAMS = {
    "init_top_range": 50.0,
    "a1_max_deg_init": 90.0,
    "init_r_lo_scale": 0.2,
    "init_r_hi_scale": 3.0,
    "o1_residual_threshold": 0.1,
    "o1_max_trials": 500,
    "o1_min_inliers": 1,
    "o2_residual_threshold": 0.2,
    "o2_max_trials": 50,
    "o2_min_inliers_per_side": 1,
    "o2_refit": False,
    "o3_residual_threshold": 0.2,
    "o3_max_trials": 50,
    "o3_min_inliers_per_side": 1,
    "o3_refit": False,
    "a1_init_residual_threshold": 0.1,
    "a1_init_ratio": 0.2,
    "a1_init_fail_tolerance": 3,
    "a2_init_residual_threshold": 0.1,
    "a2_init_ratio": 0.2,
    "a2_init_fail_tolerance": 5,
    "a_final_residual_threshold": 0.5,
    "a_final_ratio": 0.1,
    "a_final_step_deg": 0.1,
    "a_final_fail_tolerance": 10,
    "loss": "soft_l1",
    "f_scale": 0.1,
    "max_nfev": 300,
    "refine_a1_max_deg": 170.0,
    "refine_r_lo_scale": 0.1,
    "refine_r_hi_scale": 5.0,
    "fallback_a_init_shrink": 0.5,
    "refine_a1_penalty_alpha_3": 0.05,
    "refine_a2_penalty_alpha_5": 0.05,
    "refine_support_bin_deg": 1.0,
    "complexity_lambda": 0.0,
    "radius_similarity_rel_thresh": 0.0,
    "coverage_min_ratio": 0.0,
    "reject_mode": "reject",
    "reject_penalty": 1_000_000.0,
    "random_seed": 0,
}

COLOR_O1 = (0.09, 0.74, 0.81)
COLOR_O2L = (0.12, 0.47, 0.71)
COLOR_O2R = (1.0, 0.50, 0.05)
COLOR_O3L = (0.17, 0.63, 0.17)
COLOR_O3R = (0.58, 0.40, 0.74)
COLOR_JUNCTION = (1.0, 0.0, 1.0)


def _load_my_ss():
    import five_circle_core

    return five_circle_core


def _circle_polyline(center, radius, n=240):
    theta = np.linspace(0.0, 2.0 * np.pi, int(max(n, 16)), endpoint=False)
    c = np.asarray(center, dtype=np.float64)
    return c[None, :] + float(radius) * np.column_stack((np.cos(theta), np.sin(theta)))


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_debug_json(path, payload):
    if not path:
        return
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


def _compact_accuracy(acc):
    if not isinstance(acc, dict):
        return {}
    keys = ("model_mode", "mae", "rmse", "p50", "p95", "p99", "max", "count")
    return {k: acc[k] for k in keys if k in acc}


def _compact_refined(refined):
    keys = (
        "o1_center", "r1", "a1_deg",
        "o2_left_center", "o2_right_center", "r2", "a2_deg",
        "o3_left_center", "o3_right_center", "r3",
        "j1_left", "j1_right", "j2_left", "j2_right",
        "a2_unsupported_ratio",
    )
    return {k: refined[k] for k in keys if k in refined}


def _compact_initial_fit(initial_fit):
    if not isinstance(initial_fit, dict):
        return {}
    return _compact_refined(initial_fit)


def fit_five_circle_contour(points_2d, n_per_arc=160, n_circle=240, debug_path=None, debug_meta=None, **params):
    """Fit the 5-circle model and return a sampled contour plus guide circles."""
    uv = np.asarray(points_2d, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("points_2d must have shape (N, 2)")
    if len(uv) < 10:
        return uv[:0], {}
    offset = np.mean(uv, axis=0)
    uv_fit = uv - offset[None, :]

    my_ss = _load_my_ss()
    run_kwargs = dict(DEFAULT_FIVE_CIRCLE_PARAMS)
    run_kwargs.update(params)

    model5 = my_ss.run_five_circle_fit(uv_fit, **run_kwargs)
    if not model5.get("ok", False):
        raise RuntimeError(model5.get("error", "5-circle fit failed"))

    refined = model5["refined"]
    final_angles = model5.get("final_angles", {})
    a3_deg = float(final_angles.get("a3_deg", refined.get("a3_deg", 0.0)))

    profile = my_ss.build_o1_o2_o3_profile_polyline(
        o1_center=np.asarray(refined["o1_center"], dtype=np.float64),
        o1_radius=float(refined["r1"]),
        a1_deg=float(refined["a1_deg"]),
        o2_left_center=np.asarray(refined["o2_left_center"], dtype=np.float64),
        o2_right_center=np.asarray(refined["o2_right_center"], dtype=np.float64),
        o2_radius=float(refined["r2"]),
        a2_deg=float(refined["a2_deg"]),
        o3_left_center=np.asarray(refined["o3_left_center"], dtype=np.float64),
        o3_right_center=np.asarray(refined["o3_right_center"], dtype=np.float64),
        o3_radius=float(refined["r3"]),
        a3_deg=a3_deg,
        n_per_arc=int(max(n_per_arc, 8)),
        close_polygon=False,
    )
    contour = np.asarray(profile["polyline"], dtype=np.float64) + offset[None, :]

    if debug_path:
        _write_debug_json(debug_path, {
            "source": "python",
            "meta": debug_meta or {},
            "point_count": int(len(uv)),
            "offset": offset,
            "params": run_kwargs,
            "initial_fit": _compact_initial_fit(model5.get("initial_fit", {})),
            "refined": _compact_refined(refined),
            "final_angles": final_angles,
            "accuracy": _compact_accuracy(model5.get("accuracy", {})),
            "profile_fit": np.asarray(profile["polyline"], dtype=np.float64),
            "a3_deg": a3_deg,
        })

    def _offset_point(name):
        return np.asarray(refined[name], dtype=np.float64) + offset

    def _offset_profile(name):
        return np.asarray(profile[name], dtype=np.float64) + offset

    circles = [
        {
            "name": "O1",
            "center": _offset_point("o1_center"),
            "radius": float(refined["r1"]),
            "points": _circle_polyline(refined["o1_center"], refined["r1"], n_circle) + offset[None, :],
            "color": COLOR_O1,
        },
        {
            "name": "O2L",
            "center": _offset_point("o2_left_center"),
            "radius": float(refined["r2"]),
            "points": _circle_polyline(refined["o2_left_center"], refined["r2"], n_circle) + offset[None, :],
            "color": COLOR_O2L,
        },
        {
            "name": "O2R",
            "center": _offset_point("o2_right_center"),
            "radius": float(refined["r2"]),
            "points": _circle_polyline(refined["o2_right_center"], refined["r2"], n_circle) + offset[None, :],
            "color": COLOR_O2R,
        },
        {
            "name": "O3L",
            "center": _offset_point("o3_left_center"),
            "radius": float(refined["r3"]),
            "points": _circle_polyline(refined["o3_left_center"], refined["r3"], n_circle) + offset[None, :],
            "color": COLOR_O3L,
        },
        {
            "name": "O3R",
            "center": _offset_point("o3_right_center"),
            "radius": float(refined["r3"]),
            "points": _circle_polyline(refined["o3_right_center"], refined["r3"], n_circle) + offset[None, :],
            "color": COLOR_O3R,
        },
    ]
    centers = [
        {"name": "C1", "point": circles[0]["center"], "color": COLOR_O1, "size": 9.0},
        {"name": "C2L", "point": circles[1]["center"], "color": COLOR_O2L, "size": 9.0},
        {"name": "C2R", "point": circles[2]["center"], "color": COLOR_O2R, "size": 9.0},
        {"name": "C3L", "point": circles[3]["center"], "color": COLOR_O3L, "size": 9.0},
        {"name": "C3R", "point": circles[4]["center"], "color": COLOR_O3R, "size": 9.0},
    ]
    junctions = [
        {"name": name, "point": _offset_profile(name), "color": COLOR_JUNCTION, "size": 7.0}
        for name in ("j1_left", "j1_right", "j2_left", "j2_right", "j3_left", "j3_right")
        if name in profile
    ]

    info = {
        "model": "5circle",
        "result": model5,
        "refined": refined,
        "a3_deg": a3_deg,
        "circles": circles,
        "centers": centers,
        "junctions": junctions,
    }
    return contour, info
