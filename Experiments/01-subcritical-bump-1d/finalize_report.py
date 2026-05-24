"""
Auto-generate final report from sensitivity study results.

Reads results/sensitivity_results.json and fills in REPORT_template.md
to produce REPORT.md with concrete numbers and interpretations.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
REPORT_TEMPLATE = Path(__file__).parent / "REPORT_template.md"
REPORT_OUT = Path(__file__).parent / "REPORT.md"


def fmt_mm(rmse_m):
    return f"{rmse_m * 1000:.2f}"


def fmt_mm_mean_std(mean_m, std_m):
    return f"{mean_m * 1000:.2f} ± {std_m * 1000:.2f}"


def fmt_r2(r2):
    return f"{r2:.4f}"


def fmt_time(t):
    return f"{t:.1f}"


def interpret_density(results):
    """Generate interpretation text for density sweep."""
    baseline = next(r for r in results if r["density"] >= 0.99)
    baseline_rmse = baseline["zb_rmse"] * 1000

    # Find the lowest density where RMSE stays within 2x baseline
    acceptable = [r for r in results if r["zb_rmse"] * 1000 < 2 * baseline_rmse]
    min_acceptable = min(r["density"] for r in acceptable) if acceptable else None

    worst = max(results, key=lambda r: r["zb_rmse"])
    worst_rmse = worst["zb_rmse"] * 1000

    text = (
        f"At full density (100%), baseline RMSE is {baseline_rmse:.2f} mm. "
        f"The degradation with sparser observations is "
    )
    if worst_rmse / baseline_rmse < 2:
        text += f"modest — even at {worst['density']*100:.0f}% density, RMSE is only {worst_rmse:.2f} mm "
        text += f"({worst_rmse/baseline_rmse:.1f}x baseline). "
        text += "This suggests the SWE physics loss carries most of the constraint, so observations serve mainly as anchoring."
    elif worst_rmse / baseline_rmse < 5:
        text += f"moderate — at {worst['density']*100:.0f}% density, RMSE is {worst_rmse:.2f} mm "
        text += f"({worst_rmse/baseline_rmse:.1f}x baseline). "
    else:
        text += f"severe — at {worst['density']*100:.0f}% density, RMSE is {worst_rmse:.2f} mm "
        text += f"({worst_rmse/baseline_rmse:.1f}x baseline), the PINN fails to recover the bump."

    if min_acceptable is not None and min_acceptable < 1.0:
        text += f" Minimum acceptable density (< 2x baseline RMSE): **{min_acceptable*100:.0f}%**."

    return text


def interpret_noise(results):
    """Generate interpretation text for noise sweep."""
    zero_noise = next(r for r in results if r["noise_eta"] == 0)
    zero_rmse = zero_noise["zb_rmse"] * 1000

    highest = max(results, key=lambda r: r["noise_eta"])
    highest_rmse = highest["zb_rmse"] * 1000

    text = (
        f"Clean observations yield {zero_rmse:.2f} mm RMSE. "
    )

    # Find noise threshold where RMSE doubles
    acceptable = [r for r in results if r["zb_rmse"] * 1000 < 2 * zero_rmse]
    max_acceptable = max(r["noise_eta"] for r in acceptable) if acceptable else 0.0

    if highest_rmse / zero_rmse < 2:
        text += f"Inversion is robust even to the highest noise tested ({highest['noise_eta']*100:.1f}%), "
        text += f"with final RMSE only {highest_rmse:.2f} mm ({highest_rmse/zero_rmse:.1f}x clean baseline). "
        text += "The SWE physics loss effectively filters noise."
    else:
        text += f"At {highest['noise_eta']*100:.1f}% noise, RMSE reaches {highest_rmse:.2f} mm ({highest_rmse/zero_rmse:.1f}x clean baseline). "

    text += f" Maximum noise tolerance (< 2x clean RMSE): **{max_acceptable*100:.1f}%**."

    return text


def interpret_obstype(results):
    """Generate interpretation text for observation type sweep."""
    by_name = {r["obs_type"]: r for r in results}
    eta_rmse = by_name["eta only"]["zb_rmse"] * 1000
    u_rmse = by_name["u only"]["zb_rmse"] * 1000
    both_rmse = by_name["eta + u"]["zb_rmse"] * 1000

    # Identify best
    best = min(results, key=lambda r: r["zb_rmse"])

    text = (
        f"Water surface elevation alone gives {eta_rmse:.2f} mm RMSE. "
        f"Velocity alone gives {u_rmse:.2f} mm RMSE. "
        f"Combined ($\\eta + u$) gives {both_rmse:.2f} mm RMSE. "
    )

    if u_rmse < eta_rmse:
        text += (
            f"Velocity is more informative than surface elevation for this case, "
            f"consistent with Ohara 2024 and the theoretical amplification factor "
            f"$\\partial u / \\partial z_b \\propto Q/h^2$ in shallow flows. "
        )
    elif eta_rmse < u_rmse:
        text += (
            f"Surface elevation is more informative here. This contrasts with "
            f"Pujol 2025 which argued velocity is more sensitive; the difference "
            f"likely reflects the moderate Froude range (0.5-0.63) where surface "
            f"response is already substantial. "
        )

    if both_rmse < min(eta_rmse, u_rmse):
        reduction = (min(eta_rmse, u_rmse) - both_rmse) / min(eta_rmse, u_rmse) * 100
        text += (
            f"Combining both observation types reduces RMSE by {reduction:.0f}% "
            f"over the best single-type result — direct evidence that $\\eta$ and $u$ "
            f"carry complementary information about the bathymetry."
        )
    else:
        text += "Combining types does not further improve over the best single type."

    return text


def main():
    with open(RESULTS_DIR / "sensitivity_results.json") as f:
        results = json.load(f)

    # Read template
    report = REPORT_TEMPLATE.read_text()

    # --- Fill density table ---
    density_rows = []
    for r in results["density"]:
        mean_std = fmt_mm_mean_std(r["rmse_mean"], r["rmse_std"]) if "rmse_mean" in r else "n/a"
        density_rows.append(
            f"| {r['density']*100:.0f}% | {fmt_mm(r['zb_rmse'])} | {mean_std} | {fmt_r2(r['zb_r2'])} | {fmt_time(r['wall_time_s'])} |"
        )
    density_table = "\n".join(density_rows)
    density_table_placeholder = (
        "| Density | $z_b$ RMSE (mm) | R² | Training time (s) |\n"
        "|---|---|---|---|\n"
        "| 100% | | | |\n"
        "| 50% | | | |\n"
        "| 20% | | | |\n"
        "| 10% | | | |\n"
        "| 5% | | | |"
    )
    density_filled = (
        "| Density | best RMSE (mm) | mean ± std RMSE (mm) | R² | Total time (s) |\n"
        "|---|---|---|---|---|\n"
        + density_table
    )
    report = report.replace(density_table_placeholder, density_filled)

    # --- Fill noise table ---
    noise_rows = []
    for r in results["noise"]:
        mean_std = fmt_mm_mean_std(r["rmse_mean"], r["rmse_std"]) if "rmse_mean" in r else "n/a"
        noise_rows.append(
            f"| {r['noise_eta']*100:.1f}% | {fmt_mm(r['zb_rmse'])} | {mean_std} | {fmt_r2(r['zb_r2'])} | {fmt_time(r['wall_time_s'])} |"
        )
    noise_table = "\n".join(noise_rows)
    noise_table_placeholder = (
        "| Noise (% of signal) | $z_b$ RMSE (mm) | R² | Training time (s) |\n"
        "|---|---|---|---|\n"
        "| 0% | | | |\n"
        "| 1% | | | |\n"
        "| 2% | | | |\n"
        "| 5% | | | |"
    )
    noise_filled = (
        "| Noise (% of signal) | best RMSE (mm) | mean ± std RMSE (mm) | R² | Total time (s) |\n"
        "|---|---|---|---|---|\n"
        + noise_table
    )
    report = report.replace(noise_table_placeholder, noise_filled)

    # --- Fill obs type table ---
    obstype_rows = []
    for r in results["obs_type"]:
        mean_std = fmt_mm_mean_std(r["rmse_mean"], r["rmse_std"]) if "rmse_mean" in r else "n/a"
        obstype_rows.append(
            f"| {r['obs_type']} | {fmt_mm(r['zb_rmse'])} | {mean_std} | {fmt_r2(r['zb_r2'])} | {fmt_time(r['wall_time_s'])} |"
        )
    obstype_table = "\n".join(obstype_rows)
    obstype_table_placeholder = (
        "| Type | $z_b$ RMSE (mm) | R² | Training time (s) |\n"
        "|---|---|---|---|\n"
        "| $\\eta$ only | | | |\n"
        "| $u$ only | | | |\n"
        "| $\\eta + u$ | | | |"
    )
    obstype_filled = (
        "| Type | best RMSE (mm) | mean ± std RMSE (mm) | R² | Total time (s) |\n"
        "|---|---|---|---|---|\n"
        + obstype_table
    )
    report = report.replace(obstype_table_placeholder, obstype_filled)

    # --- Fill interpretations ---
    report = report.replace(
        "**Interpretation**: *[To be written after results available]*",
        "**Interpretation**: " + interpret_density(results["density"]),
        1,  # first occurrence (density)
    )
    report = report.replace(
        "**Interpretation**: *[To be written after results available]*",
        "**Interpretation**: " + interpret_noise(results["noise"]),
        1,  # next occurrence (noise)
    )
    report = report.replace(
        "**Interpretation**: *[To be written after results available]*",
        "**Interpretation**: " + interpret_obstype(results["obs_type"]),
        1,  # last occurrence (obs type)
    )

    # --- Status updated ---
    report = report.replace(
        "**Status**: Results pending sensitivity sweep completion",
        "**Status**: Complete"
    )

    # --- Cross-cutting findings section ---
    density_results = results["density"]
    noise_results = results["noise"]
    obstype_results = results["obs_type"]

    baseline = next(r for r in density_results if r["density"] >= 0.99)
    baseline_rmse = baseline["zb_rmse"] * 1000

    # Min density within 2x baseline
    acceptable_density = [r for r in density_results if r["zb_rmse"] * 1000 < 2 * baseline_rmse]
    min_density = min((r["density"] for r in acceptable_density), default=None)

    # Max noise within 2x baseline
    noise_0 = next(r for r in noise_results if r["noise_eta"] == 0)
    noise_baseline = noise_0["zb_rmse"] * 1000
    acceptable_noise = [r for r in noise_results if r["zb_rmse"] * 1000 < 2 * noise_baseline]
    max_noise = max((r["noise_eta"] for r in acceptable_noise), default=0.0)

    by_name = {r["obs_type"]: r for r in obstype_results}
    eta_rmse = by_name["eta only"]["zb_rmse"] * 1000
    u_rmse = by_name["u only"]["zb_rmse"] * 1000
    both_rmse = by_name["eta + u"]["zb_rmse"] * 1000

    cross_cutting = (
        "- **Baseline quality**: With all 500 observations, no noise, and known friction, "
        f"we recover a 200 mm bump with {baseline_rmse:.2f} mm RMSE ({baseline_rmse/200*100:.1f}% of bump height).\n"
        "- **Practical minimum observations**: "
    )
    if min_density is not None:
        cross_cutting += f"Can go as sparse as **{min_density*100:.0f}% of domain points** while keeping RMSE within 2x baseline.\n"
    else:
        cross_cutting += "Even the sparsest tested density exceeded 2x baseline RMSE.\n"

    cross_cutting += "- **Noise tolerance**: "
    if max_noise > 0:
        cross_cutting += f"Inversion remains usable up to **{max_noise*100:.1f}% noise** on surface observations.\n"
    else:
        cross_cutting += "Even 1% noise more than doubles RMSE — this case is noise-sensitive.\n"

    cross_cutting += "- **Value of velocity data**: "
    if both_rmse < eta_rmse * 0.7:
        cross_cutting += (
            f"Adding velocity reduces RMSE by {(eta_rmse - both_rmse)/eta_rmse*100:.0f}% "
            f"over $\\eta$ alone. Velocity carries complementary information about the bathymetry.\n"
        )
    elif both_rmse < eta_rmse:
        cross_cutting += (
            f"Adding velocity gives a modest {(eta_rmse - both_rmse)/eta_rmse*100:.0f}% reduction over $\\eta$ alone.\n"
        )
    else:
        cross_cutting += f"Adding velocity does not significantly improve over $\\eta$ alone in this case.\n"

    cross_cutting += (
        "- **Comparison with literature**:\n"
        "  - Ruppenthal 2026 reports RMSE robust to 5% noise using optimal control + TV regularization\n"
        "  - Liu 2024 CNN surrogate: similar sparsity tolerance but requires pretraining on 1000+ simulations\n"
    )

    # Replace the cross-cutting placeholder
    old_cross = (
        "*[Written after all sweeps complete]*\n"
        "\n"
        "- **Practical minimum observations**: What density is \"good enough\" (e.g., < 2x baseline RMSE)?\n"
        "- **Noise tolerance**: At what noise level does inversion degrade beyond useful?\n"
        "- **Value of velocity data**: How much does adding $u$ help vs $\\eta$ alone?\n"
        "- **Comparison with literature**:\n"
        "  - Ruppenthal 2026 reports RMSE robust to 5% noise using optimal control + TV regularization\n"
        "  - Liu 2024 CNN surrogate: similar sparsity tolerance but requires pretraining on 1000+ simulations"
    )
    report = report.replace(old_cross, cross_cutting)

    # Write final report
    REPORT_OUT.write_text(report)
    print(f"Report written: {REPORT_OUT}")


if __name__ == "__main__":
    main()
