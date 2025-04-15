import os
from darts.tools.plot_darts import *
from darts.tools.logging import redirect_all_output
from model_co2_spe11b import ModelCCS  # NEW
from darts.engines import redirect_darts_output
import pandas as pd
import re
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import sys
from darts.models.darts_model import DartsModel

def extract_elapsed_time(log_path, output_path):
    """
    Extracts the last occurrence of ELAPSED time from a log file and writes it to a text file.
    """
    elapsed_time = None
    try:
        with open(log_path, "r") as file:
            lines = file.readlines()

        for line in reversed(lines):
            if "ELAPSED" in line:
                parts = line.split("ELAPSED")
                if len(parts) > 1:
                    time_part = parts[1].strip().strip("()")
                    if time_part.count(":") == 2:
                        elapsed_time = time_part
                        break

        if elapsed_time:
            with open(output_path, "w") as out:
                out.write(elapsed_time)
            print(f"Elapsed time '{elapsed_time}' written to {output_path}")
        else:
            print("No elapsed time found in the log file.")

    except Exception as e:
        print(f"Error: {e}")

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()  # Ensure real-time output

    def flush(self):
        for s in self.streams:
            s.flush()

# class OutputToVTK(DartsModel):

def run(physics_type: str, case: str, out_dir: str, export_vtk=True, redirect_log=True, platform='cpu', save_ring = True, save_all_results = 1, ring_radii=[5, 10]):

    print('Platform =', platform)

    if platform == 'gpu':
        from darts.engines import set_gpu_device
        device = int(os.getenv("DEVICE", 0))
        set_gpu_device(device)  # Ensure the GPU Device Index
        print('Test started', 'physics_type:', physics_type, 'case:', case, 'platform=', platform, 'GPU device=',
              device)
    else:
        from darts.engines import set_num_threads
        NT = int(os.getenv("THREADS", 6))
        set_num_threads(NT)
        print('Test started', 'physics_type:', physics_type, 'case:', case, 'platform=', platform, 'threads=', NT)

    os.makedirs(out_dir, exist_ok=True)

    if redirect_log:
        log_path = os.path.join(out_dir, "run_n.log")
        redirect_darts_output(log_path)


    m = ModelCCS()
    m.physics_type = physics_type
    m.set_input_data(case=case)
    m.init_reservoir()
    m.init(output_folder=out_dir, platform=platform)
    m.save_data_to_h5(kind='solution')
    m.set_well_controls()

    standard_print_path = os.path.join(out_dir, "run_n_standard_print.log")
    with open(standard_print_path, "w") as std_log_file:
        tee = Tee(sys.stdout, std_log_file)
        with redirect_stdout(tee):
            ret = m.run_simulation()
            if ret != 0:
                exit(1)

    def output_to_vtk(self, ith_step: int = None, output_directory: str = None, output_properties: list = None):
        """
        Function to export results at timestamp t into `.vtk` format.

        :param ith_step: i'th reporting step
        :type ith_step: int
        :param output_directory: Name to save .vtk file
        :type output_directory: str
        :param output_properties: List of properties to include in .vtk file, default is None which will pass all
        :type output_properties: list
        """
        self.timer.node["vtk_output"].start()
        # Set default output directory
        if output_directory is None:
            output_directory = self.output_folder

        # Find index of properties to output
        ev_props = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name

        # If output_properties is None, all variables and properties from property_operators will be passed
        props_names = output_properties if output_properties is not None else list(ev_props)

        # Adding temperature and pressure to output VTK
        props_names = props_names + ['pressure'] #, 'temperature']

        timesteps, property_array = self.output_properties(output_properties=props_names, timestep=ith_step)

        # 🔍 Add this debug block right before VTK export
        print(f"[DEBUG] Exporting {len(property_array)} properties to VTK at step {ith_step}")
        print(f"[DEBUG] Property names: {list(property_array.keys())}")
        print(f"[DEBUG] Number of blocks: {len(next(iter(property_array.values()))[0])}")
        # Pass to Reservoir.output_to_vtk() method
        self.reservoir.output_to_vtk(ith_step, timesteps, output_directory, list(property_array.keys()), property_array)
        self.timer.node["vtk_output"].stop()

    def add_columns_time_data(time_data):
        molar_mass_co2 = 44.01  # kg/kmol
        time_data['Time (years)'] = time_data['time'] / 365.25
        for k in list(time_data.keys()):
            if physics_type == 'ccs' and 'V rate (m3/day)' in k:
                time_data[k.replace('V rate (m3/day)', 'V rate (kmol/day)')] = time_data[k]
                time_data[k.replace('V rate (m3/day)', 'V rate (ton/day)')] = time_data[k] * molar_mass_co2 / 1000
                time_data.drop(columns=k, inplace=True)
            if physics_type == 'ccs' and 'V  volume (m3)' in k:
                time_data[k.replace('V  volume (m3)', 'V volume (kmol)')] = time_data[k]
                time_data[k.replace('V  volume (m3)', 'V volume (Mt/year)')] = time_data[k] * molar_mass_co2 / 1000 / 1e6
                time_data.drop(columns=k, inplace=True)

    import pandas as pd
    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    add_columns_time_data(time_data)
    time_data.to_pickle(os.path.join(out_dir, 'time_data.pkl'))

    time_data_report = pd.DataFrame.from_dict(m.physics.engine.time_data_report)
    add_columns_time_data(time_data_report)
    time_data_report.to_pickle(os.path.join(out_dir, 'time_data_report.pkl'))

    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data.xlsx'))
    time_data.to_excel(writer, sheet_name='time_data')
    writer.close()

    press_gridcells = time_data_report.filter(like='reservoir').columns.tolist()
    chem_cols = time_data_report.filter(like='Kmol').columns.tolist()
    time_data_report.drop(columns=press_gridcells + chem_cols, inplace=True)
    time_data_report['Time (years)'] = time_data_report['time'] / 365.25
    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data_report.xlsx'))
    time_data_report.to_excel(writer, sheet_name='time_data_report')
    writer.close()

    m.reservoir.centers_to_vtk(out_dir)

    timing_info = m.timer.print("", "")

    print(timing_info)

    save_combined_timing_info(
        log_path=log_path,
        timing_info_str=timing_info,
        output_path=os.path.join(out_dir, "elapsed_time.txt")
    )

    # Plot convergence metrics after simulation
    if redirect_log:
        plot_convergence_metrics_from_log(log_path, out_dir, case)
        # print('yes conv plot start')
        # generate_detailed_convergence_plots_for_folder(out_dir, case)

    # if export_vtk:
    #     # read h5 file and write vtk
    #     m.reservoir.create_vtk_wells(output_directory=out_dir)
    #     for ith_step in range(len(m.idata.sim.time_steps)):
    #         m.output_to_vtk(ith_step=ith_step)

    if ring_radii[0] == 1:
        ring_radii = np.arange(1,28)

    if export_vtk:
        if len(save_all_results) == 1 and save_all_results[0] == 1:
            # read h5 file and write vtk
            m.reservoir.create_vtk_wells(output_directory=out_dir)
            for ith_step in range(len(m.idata.sim.time_steps)):
                # m.output_to_vtk(ith_step=ith_step)
                output_to_vtk(m, ith_step=ith_step)

                # if ith_step == steps_to_export[-1]:
                #     if save_ring:
                #         vtu_path = os.path.join(out_dir, f"solution_ts{ith_step}.vtu")
                #         import pyvista as pv
                #         mesh = pv.read(vtu_path)
                #         extract_and_visualize_combined_rings(model=m, mesh=mesh, output_dir=out_dir, rings=ring_radii,
                #                                              property_name="saturation")

        if len(save_all_results) == 2:
            # read h5 file and write vtk
            m.reservoir.create_vtk_wells(output_directory=out_dir)
            steps_to_export = [0] + list(range(save_all_results[0], len(m.idata.sim.time_steps), save_all_results[1])) + [len(m.idata.sim.time_steps) - 1]
            for ith_step in steps_to_export:
                output_to_vtk(m, ith_step=ith_step)
                #m.output_to_vtk(ith_step=ith_step)

                # if ith_step == steps_to_export[-1]:
                #     if save_ring:
                #         vtu_path = os.path.join(out_dir, f"solution_ts{ith_step}.vtu")
                #         import pyvista as pv
                #         mesh = pv.read(vtu_path)
                #         extract_and_visualize_combined_rings(model=m, mesh=mesh, output_dir=out_dir, rings=ring_radii,
                #                                              property_name="saturation")

        else:
            # read h5 file and write vtk
            m.reservoir.create_vtk_wells(output_directory=out_dir)
            for ith_step in range(len(m.idata.sim.time_steps)):
                # m.output_to_vtk(ith_step=ith_step)
                output_to_vtk(m, ith_step=ith_step)
            # steps_to_export = [0] + [len(m.idata.sim.time_steps) - 1]
            # for ith_step in steps_to_export:
            #     output_to_vtk(m, ith_step=ith_step)
                #m.output_to_vtk(ith_step=ith_step)

                # if ith_step == steps_to_export[-1]:
                #     if save_ring:
                #         vtu_path = os.path.join(out_dir, f"solution_ts{ith_step}.vtu")
                #         import pyvista as pv
                #         mesh = pv.read(vtu_path)
                #         extract_and_visualize_combined_rings(model=m, mesh=mesh, output_dir=out_dir, rings=ring_radii, property_name="saturation")

    # Optional: Remove large files
    for fname in ['solution.h5', 'well_data.h5']: #,'time_data.pkl', 'time_data_report.pkl']:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"Deleted {fpath}")

    return time_data, time_data_report, m.idata.well_data.wells.keys(), m.well_is_inj, m.total_poro_volume


##########################################################################################################
def plot_results(wells, well_is_inj, time_data_list, time_data_report_list,
                 label_list, physics_type, out_dir, total_poro_volume, case):  # ✅ Add case here

    plt.rc('font', size=12)

    def plot_total_inj_gas_rate_darts_volume(
            darts_df,
            style='-',
            color='#00A6D6',
            ax=None,
            alpha=1,
            total_poro_volume=None,
            out_dir=None,
            case_name="unknown"
    ):
        from scipy.integrate import trapezoid


        acc_df = pd.DataFrame()
        acc_df['time'] = darts_df['time']  # ✅ use actual time in days
        acc_df['total'] = 0.0

        # Sum all injection rates
        for col in darts_df.columns:
            if "INJ" in col and "V rate (ton/day)" in col:
                acc_df['total'] += darts_df[col]

        # Filter out early noise
        acc_df = acc_df[acc_df["time"] > 1.0]

        # Plot (convert time to years for display)
        ax = ax or plt.gca()
        ax.plot(acc_df['time'] / 365.25, acc_df['total'], style, color=color, alpha=alpha, label="total")
        ax.set_ylabel("Inj Gas Rate [Ton/Day]")
        ax.set_xlabel("Years")

        # ✅ Integration: rate (ton/day) × time (day) = total tons
        total_mass_tons = trapezoid(acc_df['total'], acc_df['time'])
        total_mass_kg = total_mass_tons * 1000
        total_mass_mt = total_mass_kg / 1e9  # convert to Mt
        print(total_mass_mt)
        # Volume estimate using CO₂ density (880 kg/m³)
        density = 880
        injected_volume_m3 = total_mass_kg / density

        annotation = f"Injected: {injected_volume_m3:,.0f} m³ CO₂"
        summary_line = f"CASE: {case_name or 'unknown'}\n"
        summary_line += f"  Total Injected Mass: {total_mass_mt:.3f} Mt\n"
        summary_line += f"  Total Injected Volume: {injected_volume_m3:,.0f} m³\n"

        # Optional pore volume filling
        percent_filled = None
        if total_poro_volume is not None:
            pore_volume = np.sum(total_poro_volume)
            percent_filled = injected_volume_m3 / pore_volume * 100
            annotation += f"\nPore Volume Filled: {percent_filled:.2f}%"
            summary_line += f"  Total Pore Volume: {pore_volume:,.0f} m³\n"
            summary_line += f"  Pore Volume Filled: {percent_filled:.2f}%\n"
        else:
            summary_line += f"  Total Pore Volume: N/A\n  Pore Volume Filled: N/A\n"

        # Annotate on plot
        ax.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        # Save summary
        if out_dir is not None:
            summary_file = os.path.join(out_dir, 'injection_summary.txt')
            with open(summary_file, 'a', encoding='utf-8') as f:
                f.write(summary_line + "\n")

        # Adjust Y axis
        ymin, ymax = ax.get_ylim()
        if ymax < 0:
            ax.set_ylim(ymin * 1.1, 0)
        if ymin > 0:
            ax.set_ylim(0, ymax * 1.1)

        # ✅ Return useful values
        return ax, {
            "total_mass_tons": total_mass_tons,
            "total_mass_mt": total_mass_mt,
            "total_volume_m3": injected_volume_m3,
            "pore_fill_percent": percent_filled,
            "summary_text": summary_line
        }

    for well_name in wells:

        ax = None
        for time_data, label in zip(time_data_list, label_list):
            # ax = plot_total_inj_gas_rate_darts_volume(time_data, ax=ax, total_poro_volume=total_poro_volume)
            ax,_ = plot_total_inj_gas_rate_darts_volume(
                time_data,
                ax=ax,
                total_poro_volume=total_poro_volume,
                out_dir=out_dir,
                case_name=case
            )
            # ax = plot_total_inj_gas_rate_darts_volume(time_data, ax=ax)  # , label=label) #NEW was ax = plot_total_inj_gas_rate_darts(time_data, ax=ax)
        ax.set(xlabel="Years", ylabel="Inj Gas Rate [Ton/Day]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'total_inj_gas_rate_' + well_name + '_' + case + '.png'))
        plt.close()

def parse_elapsed_time_txt(path):
    timing_data = {}
    try:
        with open(path, "r") as f:
            for line in f:
                match = re.match(r"\s*([\w\s<>]+)\s+([\d\.]+)\s+sec", line)
                if match:
                    label = match.group(1).strip().lower().replace(" ", "_")
                    timing_data[label] = float(match.group(2))

        # Handle broken initialization values
        if "initialization" in timing_data and timing_data["initialization"] > 86400:  # > 1 day
            total = timing_data.get("total_elapsed", 0)
            sim = timing_data.get("simulation", 0)
            newton = timing_data.get("newton_update", 0)
            vtk = timing_data.get("vtk_output", 0)
            recomputed_init = total - sim - newton - vtk
            timing_data["initialization"] = max(recomputed_init, 0)
            print(f"⚠️ Recomputed initialization time: {timing_data['initialization']:.2f} sec")

    except Exception as e:
        print(f"  ❌ Failed to read {path}: {e}")
    return timing_data

def plot_convergence_metrics_from_log(log_file_path, output_dir, case_name):
    """
    Parses and plots convergence metrics from a DARTS log file,
    using computed DT instead of logged DT.
    """
    patterns = {
        "T": r"T\s*=\s*([\d\.eE+-]+)",
        "DT": r"DT\s*=\s*([\d\.eE+-]+)",
        "NI": r"NI\s*=\s*(\d+)",
        "LI": r"LI\s*=\s*(\d+)",
        "RES": r"RES\s*=\s*([\d\.eE+-]+)",
    }

    components = [
        "initialization",
        "simulation",
        "jacobian_assembly",
        "linear_solver_solve"
    ]

    data = {key: [] for key in patterns}
    computed_dt = []
    elapsed_time = None
    timing_breakdown = None

    try:
        with open(log_file_path, "r") as f:
            prev_t = None
            for line in f:
                if "T =" in line and "CFL=" in line:
                    for key, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            val = float(match.group(1))
                            data[key].append(val)

                    # Compute DT manually from T
                    t_now = data["T"][-1]
                    if prev_t is not None:
                        computed_dt.append(t_now - prev_t)
                    else:
                        computed_dt.append(data["DT"][-1] if data["DT"] else 0.0)
                    prev_t = t_now

                if "ELAPSED" in line:
                    match = re.search(r"ELAPSED\s+(\d+):(\d+):(\d+)", line)
                    if match:
                        h, m, s = map(int, match.groups())
                        elapsed_time = h * 3600 + m * 60 + s

        # Try to load timing breakdown from elapsed_time.txt
        elapsed_txt_path = os.path.join(os.path.dirname(log_file_path), "elapsed_time.txt")
        if os.path.exists(elapsed_txt_path):
            timing_breakdown = parse_elapsed_time_txt(elapsed_txt_path)

        if not data["T"]:
            print("⚠️ No convergence data found in log file.")
            return

        plt.figure(figsize=(14, 10))

        plt.subplot(3, 2, 1)
        plt.plot(data["T"], data["NI"], label="Newton Iterations")
        plt.xlabel("Time")
        plt.ylabel("NI")
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(data["T"], data["LI"], label="Linear Iterations", color="orange")
        plt.xlabel("Time")
        plt.ylabel("LI")
        plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.plot(data["T"], data["RES"], label="Residual", color="green")
        plt.yscale("log")
        plt.xlabel("Time")
        plt.ylabel("RES")
        plt.grid(True)

        plt.subplot(3, 2, 4)
        li_ni_ratio = [li / ni if ni != 0 else 0 for li, ni in zip(data["LI"], data["NI"])]
        plt.plot(data["T"], li_ni_ratio, label="LI / NI", color="red")
        plt.xlabel("Time")
        plt.ylabel("LI / NI")
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.plot(data["T"], computed_dt, label="Computed DT", color="purple")
        plt.xlabel("Time")
        plt.ylabel("DT")
        plt.grid(True)

        plt.subplot(3, 2, 6)
        if timing_breakdown:
            x = [0]
            bottom = 0
            for comp in components:
                val = timing_breakdown.get(comp, 0)
                plt.bar(x, [val], bottom=bottom, label=comp.replace("_", " ").title())
                bottom += val
            plt.xticks([0], ["Elapsed Time"])
            plt.ylabel("Time (sec)")
            plt.title("Component-wise Elapsed Time")
            plt.legend()
        elif elapsed_time:
            elapsed_minutes = elapsed_time / 60
            plt.bar(["Elapsed Time"], [elapsed_minutes * 60], color="steelblue")
            plt.ylabel("Seconds")
            plt.title("Final Elapsed Time")
            plt.text(0, elapsed_minutes * 60, f"{elapsed_minutes:.1f} min", ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, "No ELAPSED found", ha='center', va='center')
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(f"Convergence Metrics – {case_name}", fontsize=18)

        fig_path = os.path.join(output_dir, f"convergence_metrics_{case_name}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"✅ Convergence plot saved to {fig_path}")

    except Exception as e:
        print(f"❌ Error parsing log file: {e}")

def save_combined_timing_info(log_path, timing_info_str, output_path):
    """
    Combines wall-clock elapsed time from run_n.log and detailed DARTS timing info
    into a single file at output_path.
    """
    elapsed_time_str = None
    try:
        with open(log_path, "r") as file:
            lines = file.readlines()
        for line in reversed(lines):
            if "ELAPSED" in line:
                parts = line.split("ELAPSED")
                if len(parts) > 1:
                    time_part = parts[1].strip().strip("()")
                    if time_part.count(":") == 2:
                        elapsed_time_str = time_part
                        break
    except Exception as e:
        print(f"❌ Failed to extract wall-clock time: {e}")

    try:
        with open(output_path, "w") as f:
            if elapsed_time_str:
                f.write(f"Wall-clock elapsed time (from log): {elapsed_time_str}\n\n")
            else:
                f.write("Wall-clock elapsed time not found in log.\n\n")

            f.write("Detailed timing breakdown (from simulation):\n")
            f.write(timing_info_str)

        print(f"✅ Combined timing info written to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to write timing info file: {e}")

def extract_and_visualize_combined_rings(model, mesh, output_dir, rings=[], property_name="pressure"):
    import csv
    """
    Extracts values in diamond-shaped rings (Manhattan distance) around each injector,
    and stores all data in a single CSV. Then visualizes it all in one combined plot.
    """
    if property_name == "saturation":
        property_name = "satV"

    discr = model.reservoir.discr_mesh
    get_global = discr.get_global_index
    nx, ny = discr.nx, discr.ny
    centroids = np.array([c.values for c in model.reservoir.centroids_all_cells])

    unique_ij = {(i, j) for (i, j, _) in model.well_cells}
    unique_injectors = [(i, j, 1) for (i, j) in unique_ij]
    stopped_injectors = set()
    all_entries = []

    for r in rings:
        for well_id, (i0, j0, k0) in enumerate(unique_injectors):
            if well_id in stopped_injectors:
                continue

            ring_entries = []
            has_nonzero = False

            # 🔁 Manhattan ring logic — just like original
            for di in range(-r, r + 1):
                dj = r - abs(di)
                dj_signs = [-1, 1] if dj != 0 else [1]  # prevent duplicates
                for dj_sign in dj_signs:
                    i = i0 + di
                    j = j0 + dj_sign * dj
                    k = k0

                    if not (1 <= i <= nx and 1 <= j <= ny):
                        continue

                    try:
                        g_idx = get_global(i - 1, j - 1, k - 1)
                        value = mesh.cell_data[property_name][g_idx]
                        x, y, z = centroids[g_idx]
                        ring_entries.append((well_id, i, j, k, x, y, z, r, value))
                        if value > 0:
                            has_nonzero = True
                    except Exception:
                        continue

            if not has_nonzero:
                print(f"🛑 Injector {well_id}: all values 0 in r={r}. Skipping further rings.")
                stopped_injectors.add(well_id)

            all_entries.extend(ring_entries)

    # Save single combined CSV
    if all_entries:
        out_csv = os.path.join(output_dir, f"diamond_rings_combined_ijk_{property_name}.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["well_id", "i", "j", "k", "x", "y", "z", "ring", property_name])
            writer.writerows(all_entries)
        print(f"✅ Combined CSV saved: {out_csv}")

        # Plot
        df = pd.read_csv(out_csv)
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(df["x"], df["y"], c=df[property_name], cmap="viridis", s=80)
        plt.colorbar(sc, label=property_name)

        for well_id, group in df.groupby("well_id"):
            inj_x = group[group["i"] == group["i"].median()]["x"].iloc[0]
            inj_y = group[group["j"] == group["j"].median()]["y"].iloc[0]
            plt.scatter(inj_x, inj_y, color="red", s=120, marker="*", label=f"Injector {well_id}")

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.grid(True)
        plotted_rings = sorted(df["ring"].unique())
        plt.title(f"{property_name} in Diamond Rings {plotted_rings} (All Injectors)")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"diamond_rings_all_{property_name}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Combined plot saved: {plot_path}")
    else:
        print("❌ No data to save or visualize.")

if __name__ == '__main__':
    platform = 'gpu' if os.getenv('TEST_GPU') == '1' else 'cpu'

    physics_list = ['ccs']

    # #if platform == 'gpu': #Deactivated for now to make model more userfriendly for inspection
    # if platform == 'gpu':
    #     cases_array = np.load("cases_array.npy", allow_pickle=True)
    #     case_index = os.getenv("CASE_INDEX")
    #     case_name = os.getenv("CASE_NAME")

    #     if case_index is not None:
    #         import ast
    #         case_index = ast.literal_eval(case_index)
    #         if isinstance(case_index, int):
    #             cases_list = [(case_index, cases_array[case_index])]
    #         elif isinstance(case_index, list):
    #             cases_list = [(i, cases_array[i]) for i in case_index]
    #     elif case_name is not None:
    #         cases_list = [(i, c) for i, c in enumerate(cases_array) if case_name in c]
    #     else:
    #         n_splits = int(os.getenv("SPLITS", 3))
    #         split_id = int(os.getenv("ID", 0))
    #         chunks = np.array_split(list(enumerate(cases_array)), n_splits)
    #         cases_list = chunks[split_id]

    # else:
    cases_list = []
    #cases_list += ['fault=FM1_cut=CO1_grid=G1_top=TS2_mod=OBJ_mult=1']
    cases_list += ['grid_CCS_maarten']  # Add more if needed
    #cases_list += ['grid_CCS_maarten_homogeneous']
    # cases_list += ['grid_CCS_maarten_homogeneous']  # Homogeneous, one region assignment

    #cases_list += ["case_1_50x50x40"]
    #cases_list += ["case_1_100x100x80"]
    #cases_list += ["case_1_125x125x80"]
    # cases_list += ["case_1_250x250x80"]
    # cases_list += ["case_2_50x50x40"]
    # cases_list += ["case_2_100x100x80"]
    # cases_list += ["case_2_125x125x80"]
    # cases_list += ["case_2_250x250x80"]
    # cases_list += ["case_3_50x50x40"]
    # cases_list += ["case_3_100x100x80"]
    # cases_list += ["case_3_125x125x80"]
    # cases_list += ["case_3_250x250x80"]

    # Wrap CPU cases with fake index so main loop always works
    cases_list = [(i, case) for i, case in enumerate(cases_list)]


    well_controls = []
    well_controls +=  ['wbhp']
    #well_controls += ['rate']


    for physics_type in physics_list:
        for case_idx, case_geom in cases_list:
        # for case_geom in cases_list:
            for wctrl in well_controls:
                if physics_type == 'deadoil' and wctrl == 'wrate':
                    continue

                tag = f"{int(case_idx):03d}"

                case = case_geom + '_' + wctrl
                folder_name = 'results_' + physics_type + '_' + case + '_' + tag + "_BHP_Check_gradientactivated_2"
                out_dir = os.path.join("results", folder_name)

                time_data, time_data_report, wells, well_is_inj, total_poro_volume = run(
                    physics_type=physics_type,
                    case=case,
                    out_dir=out_dir,
                    redirect_log=True,
                    export_vtk=True,
                    platform=platform,
                    save_ring=True,
                    save_all_results = [1500], #1 #[3,4] #1 => saves vtk's from all years, 2 => saves vtk's around the so many years (i.e. [2,3]), 3 => only saves the first and last vtk (Actually true for anything except 1 or [.,.])
                    ring_radii= [1] #[5, 10, 15]
                )

                # one can read well results from pkl file to add/change well plots without re-running the model
                pkl1_dir = '..'
                pkl_fname = 'time_data.pkl'
                pkl_report_fname = 'time_data_report.pkl'
                time_data_list = [time_data]
                time_data_report_list = [time_data_report]
                label_list = [None]

                # compare the current results with another run
                # pkl1_dir = r'../../../open-darts_dev/models/cpg_sloping_fault/results_' + physics_type + '_' + case_geom
                # time_data_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_fname))
                # time_data_report_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_report_fname))
                # time_data_list = [time_data_1, time_data]
                # time_data_report_list = [time_data_report_1, time_data_report]
                # label_list = ['1', 'current']

                plot_results(wells=wells, well_is_inj=well_is_inj,
                             time_data_list=time_data_list, time_data_report_list=time_data_report_list,
                             label_list=label_list,
                             physics_type=physics_type, out_dir=out_dir,
                             total_poro_volume=total_poro_volume,
                             case=case)  # ✅ Add this

                # plot_results(wells=wells, well_is_inj=well_is_inj,
                #              time_data_list=time_data_list, time_data_report_list=time_data_report_list,
                #              label_list=label_list,
                #              physics_type=physics_type, out_dir=out_dir,
                #              total_poro_volume=total_poro_volume)

                # plot_results(wells=wells, well_is_inj=well_is_inj,
                #              time_data_list=time_data_list, time_data_report_list=time_data_report_list,
                #              label_list=label_list,
                #              physics_type=physics_type, out_dir=out_dir)


